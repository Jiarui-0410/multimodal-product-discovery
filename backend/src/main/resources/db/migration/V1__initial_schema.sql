CREATE TABLE users (
    user_id BIGSERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    ml_index_id BIGINT NOT NULL UNIQUE,
    name VARCHAR(500) NOT NULL,
    category VARCHAR(150),
    color VARCHAR(100),
    season VARCHAR(100),
    usage VARCHAR(100),
    gender VARCHAR(100),
    image_path VARCHAR(1000),
    semantic_embedding TEXT
);

CREATE TABLE interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_interaction_type CHECK (
        interaction_type IN ('VIEW', 'CLICK', 'LIKE', 'SAVE', 'PURCHASE')
    )
);

CREATE TABLE search_history (
    search_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    query_text VARCHAR(1000),
    query_type VARCHAR(20) NOT NULL,
    searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_query_type CHECK (query_type IN ('TEXT', 'IMAGE'))
);

CREATE TABLE user_profiles (
    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    category_preferences TEXT NOT NULL DEFAULT '{}',
    color_preferences TEXT NOT NULL DEFAULT '{}',
    usage_preferences TEXT NOT NULL DEFAULT '{}',
    semantic_embedding TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_interactions_user_time ON interactions(user_id, occurred_at DESC);
CREATE INDEX idx_interactions_product ON interactions(product_id);
CREATE INDEX idx_search_history_user_time ON search_history(user_id, searched_at DESC);
