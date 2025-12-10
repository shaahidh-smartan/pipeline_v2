-- ============================================================================
-- BPBreID Embeddings Table Setup (Per-Part Storage)
-- ============================================================================
-- This SQL script creates the bpbreid_embeddings table for storing person
-- re-identification embeddings with SEPARATE rows for each body part.
--
-- Strategy: Store each body part as a separate row with the same face_id (PID)
-- Benefits:
--   1. Can use IVFFlat (512 dims << 2000 limit)
--   2. Query specific body parts (e.g., only upper body)
--   3. Filter by visibility before matching
--   4. Part-level similarity matching
--
-- Run this BEFORE starting the bpbreid_embedding_collector.py script
-- ============================================================================

-- Create the bpbreid_embeddings table (one row per body part)
CREATE TABLE IF NOT EXISTS bpbreid_embeddings (
    id SERIAL PRIMARY KEY,
    face_id INTEGER NOT NULL REFERENCES face_embeddings(id) ON DELETE CASCADE,
    body_part_id INTEGER NOT NULL,     -- 0=head, 1=upper_torso, 2=mid_torso, 3=lower_torso, 4=legs, 5=background
    embedding vector(512) NOT NULL,     -- Single body part embedding (512 dims)
    visibility FLOAT NOT NULL,          -- Visibility score for this part (0.0 to 1.0)
    camera_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure we don't duplicate body parts for same detection
    UNIQUE(face_id, body_part_id, camera_id, created_at)
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_bpbreid_face_id ON bpbreid_embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_bpbreid_body_part ON bpbreid_embeddings(body_part_id);
CREATE INDEX IF NOT EXISTS idx_bpbreid_camera_id ON bpbreid_embeddings(camera_id);
CREATE INDEX IF NOT EXISTS idx_bpbreid_created_at ON bpbreid_embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_bpbreid_visibility ON bpbreid_embeddings(visibility);

-- Create vector similarity search index using IVFFlat (efficient for 512 dims)
-- IVFFlat is perfect since each part is 512D (well under 2000D limit)
CREATE INDEX IF NOT EXISTS idx_bpbreid_embedding_vector
ON bpbreid_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Add table and column comments for documentation
COMMENT ON TABLE bpbreid_embeddings IS 'BPBreID person re-identification embeddings - one row per body part';
COMMENT ON COLUMN bpbreid_embeddings.face_id IS 'Foreign key to face_embeddings table - links person identity (PID)';
COMMENT ON COLUMN bpbreid_embeddings.body_part_id IS 'Body part index: 0=head, 1=upper_torso, 2=mid_torso, 3=lower_torso, 4=legs, 5=background';
COMMENT ON COLUMN bpbreid_embeddings.embedding IS 'Single body part embedding (512 dimensions)';
COMMENT ON COLUMN bpbreid_embeddings.visibility IS 'Visibility score for this body part (0.0 = occluded, 1.0 = fully visible)';
COMMENT ON COLUMN bpbreid_embeddings.camera_id IS 'Camera identifier (left, right, center, back)';

-- ============================================================================
-- Body Part Reference
-- ============================================================================
-- body_part_id = 0: Head/Upper body
-- body_part_id = 1: Upper Torso
-- body_part_id = 2: Middle Torso
-- body_part_id = 3: Lower Torso
-- body_part_id = 4: Upper Legs
-- body_part_id = 5: Lower/Background

-- ============================================================================
-- Verify table creation
-- ============================================================================
-- Run this to check if table was created successfully:
-- SELECT table_name, column_name, data_type
-- FROM information_schema.columns
-- WHERE table_name = 'bpbreid_embeddings';

-- ============================================================================
-- Example queries
-- ============================================================================

-- Get all embeddings for a specific person (by face_id)
-- SELECT b.*, f.person_name
-- FROM bpbreid_embeddings b
-- JOIN face_embeddings f ON b.face_id = f.id
-- WHERE b.face_id = 1
-- ORDER BY body_part_id;

-- Count embeddings per person per body part
-- SELECT f.person_name, f.id as face_id, b.body_part_id, COUNT(b.id) as embedding_count
-- FROM face_embeddings f
-- LEFT JOIN bpbreid_embeddings b ON f.id = b.face_id
-- GROUP BY f.id, f.person_name, b.body_part_id
-- ORDER BY f.person_name, b.body_part_id;

-- Find similar persons using ONLY VISIBLE body parts (visibility >= 0.5)
-- Query all 6 parts and aggregate the scores
-- WITH visible_matches AS (
--     SELECT f.person_name, b.body_part_id, b.camera_id,
--            1 - (b.embedding <=> %s::vector) AS similarity
--     FROM bpbreid_embeddings b
--     JOIN face_embeddings f ON b.face_id = f.id
--     WHERE b.body_part_id = %s  -- Query one part at a time
--       AND b.visibility >= 0.5  -- Only use visible parts
--     ORDER BY b.embedding <=> %s::vector ASC
--     LIMIT 10
-- )
-- SELECT person_name, AVG(similarity) as avg_similarity, COUNT(*) as matched_parts
-- FROM visible_matches
-- GROUP BY person_name
-- ORDER BY avg_similarity DESC;

-- Find matches for a specific body part (e.g., upper torso)
-- SELECT f.person_name, b.camera_id, b.visibility,
--        1 - (b.embedding <=> %s::vector) AS similarity
-- FROM bpbreid_embeddings b
-- JOIN face_embeddings f ON b.face_id = f.id
-- WHERE b.body_part_id = 1  -- Upper torso
--   AND b.visibility >= 0.7
-- ORDER BY b.embedding <=> %s::vector ASC
-- LIMIT 10;

-- ============================================================================
