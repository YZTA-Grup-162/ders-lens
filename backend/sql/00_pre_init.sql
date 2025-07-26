-- Script to be run before the main initialization
-- This ensures the user exists without causing errors
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'attention_user') THEN
        CREATE USER attention_user WITH PASSWORD 'attention_pass';
        RAISE NOTICE 'Created user attention_user';
    ELSE
        RAISE NOTICE 'User attention_user already exists';
    END IF;
END
$$;
