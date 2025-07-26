-- This script is idempotent and can be run multiple times without errors

-- Connect to the default database first
\c postgres

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE attentionpulse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'attentionpulse')\gexec

-- Connect to the attentionpulse database
\c attentionpulse

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create enum types if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('student', 'teacher', 'admin');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'feedback_type') THEN
        CREATE TYPE feedback_type AS ENUM ('attention', 'engagement', 'general');
    END IF;
END
$$;

-- Create tables if they don't exist
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role user_role NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    teacher_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    session_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMPTZ DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attention_scores (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    attention_level FLOAT NOT NULL,
    engagement_level FLOAT,
    distraction_type VARCHAR(50),
    confidence FLOAT NOT NULL,
    face_detected BOOLEAN DEFAULT false,
    head_pose_x FLOAT,
    head_pose_y FLOAT,
    head_pose_z FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    feedback_type feedback_type NOT NULL,
    message TEXT NOT NULL,
    suggestions TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_attention_scores_session_id ON attention_scores(session_id);
CREATE INDEX IF NOT EXISTS idx_attention_scores_timestamp ON attention_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_student_id ON sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_sessions_teacher_id ON sessions(teacher_id);

-- Create or replace function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create or replace triggers for updated_at
DO $$
BEGIN
    -- Drop existing triggers if they exist
    DROP TRIGGER IF EXISTS update_users_updated_at ON users;
    DROP TRIGGER IF EXISTS update_sessions_updated_at ON sessions;
    
    -- Recreate triggers
    CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
END
$$;

-- Create default admin user if it doesn't exist (password: admin123)
INSERT INTO users (email, username, full_name, hashed_password, role, is_active)
SELECT 'admin@derslens.com', 'admin', 'System Administrator', 
       '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'admin', true
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'admin@derslens.com');

-- Grant permissions to attention_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO attention_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO attention_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO attention_user;
