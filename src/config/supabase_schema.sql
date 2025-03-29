-- Supabase SQL Schema for Wearable Data Insight Generator

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pgcrypto for password hashing
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable RLS (Row Level Security)
ALTER DATABASE postgres SET "app.jwt_secret" TO 'your-jwt-secret-here';

-- Create custom types
CREATE TYPE user_activity_level AS ENUM ('sedentary', 'light', 'moderate', 'active', 'athlete');
CREATE TYPE insight_type AS ENUM ('recovery', 'sleep', 'activity', 'strain', 'nutrition', 'stress', 'general', 'warning', 'recommendation');
CREATE TYPE device_type AS ENUM ('fitbit', 'garmin', 'whoop', 'apple_health', 'google_fit', 'oura', 'other');
CREATE TYPE auth_provider AS ENUM ('email', 'google', 'apple', 'fitbit', 'garmin', 'whoop');

-- Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    gender TEXT,
    height NUMERIC,
    weight NUMERIC,
    activity_level user_activity_level DEFAULT 'moderate',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    avatar_url TEXT,
    preferences JSONB DEFAULT '{}'::jsonb,
    
    -- Ensure email is valid format
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+[.][A-Za-z]+$')
);

-- Enable RLS on users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create policy for users table
CREATE POLICY "Users can only access their own data" 
    ON users FOR ALL 
    USING (auth.uid() = id);

-- Connected Devices Table
CREATE TABLE connected_devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_type device_type NOT NULL,
    device_name TEXT NOT NULL,
    device_id TEXT,
    last_synced TIMESTAMP WITH TIME ZONE,
    auth_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sync_settings JSONB DEFAULT '{}'::jsonb,
    
    -- Each user can only have one active device of each type
    CONSTRAINT unique_active_device_per_user UNIQUE (user_id, device_type, is_active)
);

-- Enable RLS on connected_devices table
ALTER TABLE connected_devices ENABLE ROW LEVEL SECURITY;

-- Create policy for connected_devices table
CREATE POLICY "Users can only access their own devices" 
    ON connected_devices FOR ALL 
    USING (auth.uid() = user_id);

-- Wearable Metrics Table
CREATE TABLE wearable_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id UUID REFERENCES connected_devices(id) ON DELETE SET NULL,
    date DATE NOT NULL,
    hrv_rmssd NUMERIC,
    resting_hr NUMERIC,
    sleep_hours NUMERIC,
    sleep_quality NUMERIC,
    recovery_score NUMERIC,
    strain NUMERIC,
    steps INTEGER,
    active_minutes INTEGER,
    calories NUMERIC,
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Each user can only have one set of metrics per day
    CONSTRAINT unique_metrics_per_day UNIQUE (user_id, date)
);

-- Enable RLS on wearable_metrics table
ALTER TABLE wearable_metrics ENABLE ROW LEVEL SECURITY;

-- Create policy for wearable_metrics table
CREATE POLICY "Users can only access their own metrics" 
    ON wearable_metrics FOR ALL 
    USING (auth.uid() = user_id);

-- Insights Table
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    type insight_type NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    content TEXT,
    importance INTEGER DEFAULT 1,
    is_read BOOLEAN DEFAULT FALSE,
    is_actionable BOOLEAN DEFAULT FALSE,
    action_taken BOOLEAN DEFAULT FALSE,
    related_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on insights table
ALTER TABLE insights ENABLE ROW LEVEL SECURITY;

-- Create policy for insights table
CREATE POLICY "Users can only access their own insights" 
    ON insights FOR ALL 
    USING (auth.uid() = user_id);

-- Training Plans Table
CREATE TABLE training_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    plan_data JSONB NOT NULL
);

-- Enable RLS on training_plans table
ALTER TABLE training_plans ENABLE ROW LEVEL SECURITY;

-- Create policy for training_plans table
CREATE POLICY "Users can only access their own training plans" 
    ON training_plans FOR ALL 
    USING (auth.uid() = user_id);

-- Visualization Settings Table
CREATE TABLE visualization_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    settings JSONB NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Each user can only have one default setting per visualization type
    CONSTRAINT unique_default_per_type UNIQUE (user_id, type, is_default)
);

-- Enable RLS on visualization_settings table
ALTER TABLE visualization_settings ENABLE ROW LEVEL SECURITY;

-- Create policy for visualization_settings table
CREATE POLICY "Users can only access their own visualization settings" 
    ON visualization_settings FOR ALL 
    USING (auth.uid() = user_id);

-- Conversation History Table
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on conversation_history table
ALTER TABLE conversation_history ENABLE ROW LEVEL SECURITY;

-- Create policy for conversation_history table
CREATE POLICY "Users can only access their own conversation history" 
    ON conversation_history FOR ALL 
    USING (auth.uid() = user_id);

-- Create indexes for performance
CREATE INDEX idx_wearable_metrics_user_date ON wearable_metrics(user_id, date);
CREATE INDEX idx_insights_user_date ON insights(user_id, date);
CREATE INDEX idx_connected_devices_user ON connected_devices(user_id);
CREATE INDEX idx_training_plans_user ON training_plans(user_id);
CREATE INDEX idx_conversation_history_session ON conversation_history(session_id);

-- Create functions for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at timestamps
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_connected_devices_updated_at
    BEFORE UPDATE ON connected_devices
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_wearable_metrics_updated_at
    BEFORE UPDATE ON wearable_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_insights_updated_at
    BEFORE UPDATE ON insights
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_plans_updated_at
    BEFORE UPDATE ON training_plans
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_visualization_settings_updated_at
    BEFORE UPDATE ON visualization_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
