-- Supabase Schema Enhancements for Wearable Data Insight Generator

-- Add missing fields to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS training_personality TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS goals JSONB DEFAULT '{}'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS baseline_metrics JSONB DEFAULT '{}'::jsonb;

-- Add detailed user profile table for more extensive profile data
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fitness_level INTEGER, -- 1-10 scale
    training_history TEXT,
    medical_conditions JSONB,
    supplements JSONB,
    nutrition_preferences JSONB,
    sleep_preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_user_profile UNIQUE (user_id)
);

-- Enable RLS on user_profiles table
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Create policy for user_profiles table
CREATE POLICY "Users can only access their own profile data" 
    ON user_profiles FOR ALL 
    USING (auth.uid() = user_id);

-- Add simulation scenarios table
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    scenario_type TEXT NOT NULL,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on simulation_scenarios table
ALTER TABLE simulation_scenarios ENABLE ROW LEVEL SECURITY;

-- Create policy for simulation_scenarios table
CREATE POLICY "Users can only access their own simulation scenarios" 
    ON simulation_scenarios FOR ALL 
    USING (auth.uid() = user_id);

-- Add simulation results table
CREATE TABLE IF NOT EXISTS simulation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scenario_id UUID REFERENCES simulation_scenarios(id) ON DELETE CASCADE,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    metrics_summary JSONB,
    insights_generated INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on simulation_results table
ALTER TABLE simulation_results ENABLE ROW LEVEL SECURITY;

-- Create policy for simulation_results table
CREATE POLICY "Users can only access their own simulation results" 
    ON simulation_results FOR ALL 
    USING (auth.uid() = user_id);

-- Add timeline visualizations table
CREATE TABLE IF NOT EXISTS timeline_visualizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    metrics_included TEXT[],
    insight_types_included TEXT[],
    visualization_config JSONB,
    html_content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on timeline_visualizations table
ALTER TABLE timeline_visualizations ENABLE ROW LEVEL SECURITY;

-- Create policy for timeline_visualizations table
CREATE POLICY "Users can only access their own timeline visualizations" 
    ON timeline_visualizations FOR ALL 
    USING (auth.uid() = user_id);

-- Add OAuth provider configurations table
CREATE TABLE IF NOT EXISTS oauth_provider_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider device_type NOT NULL,
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    scopes TEXT[],
    additional_params JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_provider UNIQUE (provider)
);

-- Add raw wearable data table for storing original device data
CREATE TABLE IF NOT EXISTS raw_wearable_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id UUID REFERENCES connected_devices(id) ON DELETE SET NULL,
    data_type TEXT NOT NULL, -- e.g., 'heart_rate', 'sleep', 'activity'
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    raw_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on raw_wearable_data table
ALTER TABLE raw_wearable_data ENABLE ROW LEVEL SECURITY;

-- Create policy for raw_wearable_data table
CREATE POLICY "Users can only access their own raw data" 
    ON raw_wearable_data FOR ALL 
    USING (auth.uid() = user_id);

-- Add user feedback table for insights
CREATE TABLE IF NOT EXISTS insight_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    rating INTEGER, -- 1-5 scale
    feedback_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_user_insight_feedback UNIQUE (user_id, insight_id)
);

-- Enable RLS on insight_feedback table
ALTER TABLE insight_feedback ENABLE ROW LEVEL SECURITY;

-- Create policy for insight_feedback table
CREATE POLICY "Users can only access their own insight feedback" 
    ON insight_feedback FOR ALL 
    USING (auth.uid() = user_id);

-- Add intraday metrics table for high-resolution data
CREATE TABLE IF NOT EXISTS intraday_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id UUID REFERENCES connected_devices(id) ON DELETE SET NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_type TEXT NOT NULL, -- e.g., 'heart_rate', 'steps', 'calories'
    value NUMERIC NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Index for efficient time-series queries
    CONSTRAINT unique_user_metric_timestamp UNIQUE (user_id, metric_type, timestamp)
);

-- Enable RLS on intraday_metrics table
ALTER TABLE intraday_metrics ENABLE ROW LEVEL SECURITY;

-- Create policy for intraday_metrics table
CREATE POLICY "Users can only access their own intraday metrics" 
    ON intraday_metrics FOR ALL 
    USING (auth.uid() = user_id);

-- Create indexes for performance on new tables
CREATE INDEX idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX idx_simulation_scenarios_user_id ON simulation_scenarios(user_id);
CREATE INDEX idx_simulation_results_user_scenario ON simulation_results(user_id, scenario_id);
CREATE INDEX idx_timeline_visualizations_user_dates ON timeline_visualizations(user_id, start_date, end_date);
CREATE INDEX idx_raw_wearable_data_user_type ON raw_wearable_data(user_id, data_type);
CREATE INDEX idx_insight_feedback_insight_id ON insight_feedback(insight_id);
CREATE INDEX idx_intraday_metrics_user_timestamp ON intraday_metrics(user_id, timestamp);
CREATE INDEX idx_intraday_metrics_metric_type ON intraday_metrics(metric_type);
