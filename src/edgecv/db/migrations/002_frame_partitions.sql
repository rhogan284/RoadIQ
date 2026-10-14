-- W8 milestone: frames is partitioned in practice, not only in its DDL.
--
-- 001 left frames with a DEFAULT partition only, on the reasoning that "DEFAULT routes
-- every row correctly until then". It does, and that is the problem: PG16 §5.11 --
-- creating a partition while a DEFAULT exists scans the DEFAULT under ACCESS EXCLUSIVE
-- and fails if it holds any row that belongs to the new partition. Every row DEFAULT
-- caught sits in the way of the partition that should hold it (T2, 9 Sep:
-- "updated partition constraint for default partition ... would be violated").
--
-- So monthly partitions are created ahead of the data, and DEFAULT is a safety net that
-- should stay empty. Anything found in it is moved out the way that was proven on
-- 9 Sep: DETACH the default, create the partitions, re-insert through the parent so
-- each row routes by captured_at, empty the default, ATTACH it back.
--
-- No foreign key references frames, so DETACH is not blocked. Runs as one statement,
-- so on an autocommit connection it is still atomic; concurrent inserts wait on the lock.

CREATE OR REPLACE FUNCTION ensure_frame_partitions(months_ahead integer DEFAULT 3)
RETURNS bigint
LANGUAGE plpgsql
AS $$
DECLARE
    this_month  date := date_trunc('month', now())::date;
    first_month date;
    last_month  date;
    stranded    bigint;
    m           date;
BEGIN
    SELECT count(*),
           least(date_trunc('month', min(captured_at))::date, this_month),
           greatest(date_trunc('month', max(captured_at))::date,
                    (this_month + make_interval(months => months_ahead))::date)
      INTO stranded, first_month, last_month
      FROM frames_default;

    IF stranded > 0 THEN
        ALTER TABLE frames DETACH PARTITION frames_default;
    END IF;

    FOR m IN SELECT generate_series(first_month, last_month, interval '1 month')::date LOOP
        IF to_regclass(format('frames_%s', to_char(m, 'YYYY_MM'))) IS NULL THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF frames FOR VALUES FROM (%L) TO (%L)',
                format('frames_%s', to_char(m, 'YYYY_MM')),
                m, (m + interval '1 month')::date);
        END IF;
    END LOOP;

    IF stranded > 0 THEN
        INSERT INTO frames SELECT * FROM frames_default;
        TRUNCATE frames_default;
        ALTER TABLE frames ATTACH PARTITION frames_default DEFAULT;
    END IF;

    RETURN stranded;   -- rows rescued from DEFAULT; 0 is the healthy answer
END;
$$;

SELECT ensure_frame_partitions();
