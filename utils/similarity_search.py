import numpy as np
from .database_manager import DatabaseManager


class SimilaritySearch:
    """Vector similarity search operations for face and person embeddings."""
    
    def __init__(self, database_manager=None):
        self.db_manager = database_manager or DatabaseManager()
    
    def find_face_match_cosine(self, query_embedding, threshold=0.6):
        """Find face match using cosine similarity."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None, 0.0
            vec = vec / norm
            
            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0
            
            cur = conn.cursor()
            
            # Cosine similarity search
            cur.execute("""
                SELECT person_name, 1 - (embedding <=> %s::vector) AS similarity
                FROM face_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1;
            """, (vec.tolist(), vec.tolist(), threshold, vec.tolist()))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                person_name, similarity = row
                return {
                    'name': person_name,
                    'similarity': float(similarity),
                    'match_type': 'cosine'
                }, float(similarity)
            
            return None, 0.0
            
        except Exception as e:
            print(f"[ERROR] Face cosine search failed: {e}")
            return None, 0.0
    
    def find_face_match_euclidean(self, query_embedding, threshold=0.8):
        """Find face match using Euclidean distance."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None, 0.0
            vec = vec / norm
            
            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0
            
            cur = conn.cursor()
            
            # Euclidean distance search - INCLUDE ID for BPBreID foreign key
            cur.execute("""
                SELECT id, person_name, embedding <-> %s::vector AS distance
                FROM face_embeddings
                WHERE embedding <-> %s::vector <= %s
                ORDER BY distance ASC
                LIMIT 1;
            """, (vec.tolist(), vec.tolist(), threshold))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                face_id, person_name, distance = row
                similarity = 1.0 / (1.0 + distance)  # Convert to similarity

                return {
                    'id': face_id,  # ADD face_id for BPBreID foreign key
                    'name': person_name,
                    'similarity': float(similarity),
                    'match_type': 'euclidean',
                    'distance': float(distance)
                }, float(similarity)
            
            return None, 0.0
            
        except Exception as e:
            print(f"[ERROR] Face euclidean search failed: {e}")
            return None, 0.0
    
    def find_person_match_cosine(self, query_embedding, threshold=0.8):
        """Find person match using cosine similarity."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None, 0.0
            vec = vec / norm
            
            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT person_name, camera_id, 1 - (embedding <=> %s::vector) AS similarity
                FROM vgg_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1;
            """, (vec.tolist(), vec.tolist(), threshold, vec.tolist()))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                person_name, source_camera, similarity = row
                return {
                    'person_name': person_name,
                    'similarity': float(similarity),
                    'source_camera': source_camera,
                    'match_type': 'cosine'
                }, float(similarity)
            
            return None, 0.0
            
        except Exception as e:
            print(f"[ERROR] Person cosine search failed: {e}")
            return None, 0.0
    
    def find_person_match_euclidean(self, query_embedding, threshold=0.9):
        """Find person match using Euclidean distance with fixed similarity conversion."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None, 0.0
            vec = vec / norm
            
            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0
            
            cur = conn.cursor()
            
            cur.execute("""
                SELECT person_name, camera_id, embedding <-> %s::vector AS distance
                FROM vgg_embeddings
                ORDER BY distance ASC
                LIMIT 1;
            """, (vec.tolist(),))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                person_name, source_camera, distance = row
                
                # Better conversion for normalized vectors
                similarity = max(0.0, 1.0 - (distance / 2.0))
                
                # Use threshold of 0.6 for this scale (equivalent to 0.8 cosine similarity)
                if similarity >= 0.6:
                    return {
                        'person_name': person_name,
                        'similarity': float(similarity),
                        'source_camera': source_camera,
                        'match_type': 'euclidean',
                        'distance': float(distance)
                    }, float(similarity)
                else:
                    return None, float(similarity)
            
            return None, 0.0
            
        except Exception as e:
            print(f"[ERROR] Person euclidean search failed: {e}")
            return None, 0.0
    
    def compare_embeddings_cosine(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Calculate cosine similarity
            similarity = float(np.dot(embedding1_norm, embedding2_norm))
            return similarity
            
        except Exception as e:
            print(f"[ERROR] Cosine similarity calculation failed: {e}")
            return 0.0
    
    def compare_embeddings_euclidean(self, embedding1, embedding2):
        """Calculate Euclidean distance between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return float('inf')
            
            embedding1_norm = embedding1 / norm1
            embedding2_norm = embedding2 / norm2
            
            # Calculate Euclidean distance
            distance = float(np.linalg.norm(embedding1_norm - embedding2_norm))
            return distance
            
        except Exception as e:
            print(f"[ERROR] Euclidean distance calculation failed: {e}")
            return float('inf')
    
    def get_top_matches(self, query_embedding, table='face_embeddings', 
                       similarity_type='cosine', top_k=5):
        """Get top K matches for a query embedding."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return []
            vec = vec / norm
            
            conn = self.db_manager.get_connection()
            if not conn:
                return []
            
            cur = conn.cursor()
            
            if similarity_type == 'cosine':
                if table == 'face_embeddings':
                    cur.execute("""
                        SELECT person_name, 1 - (embedding <=> %s::vector) AS similarity
                        FROM face_embeddings
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s;
                    """, (vec.tolist(), vec.tolist(), top_k))
                else:  # vgg_embeddings
                    cur.execute("""
                        SELECT person_name, camera_id, 1 - (embedding <=> %s::vector) AS similarity
                        FROM vgg_embeddings
                        ORDER BY embedding <=> %s::vector ASC
                        LIMIT %s;
                    """, (vec.tolist(), vec.tolist(), top_k))
            else:  # euclidean
                if table == 'face_embeddings':
                    cur.execute("""
                        SELECT person_name, embedding <-> %s::vector AS distance
                        FROM face_embeddings
                        ORDER BY distance ASC
                        LIMIT %s;
                    """, (vec.tolist(), top_k))
                else:  # vgg_embeddings
                    cur.execute("""
                        SELECT person_name, camera_id, embedding <-> %s::vector AS distance
                        FROM vgg_embeddings
                        ORDER BY distance ASC
                        LIMIT %s;
                    """, (vec.tolist(), top_k))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return results

        except Exception as e:
            print(f"[ERROR] Top matches search failed: {e}")
            return []

    def find_solider_match_cosine(self, query_embedding, threshold=0.7):
        """Find person match using SOLIDER embeddings (1024-dim) with cosine similarity."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None, 0.0
            vec = vec / norm

            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0

            cur = conn.cursor()

            cur.execute("""
                SELECT person_name, camera_id, 1 - (embedding <=> %s::vector) AS similarity
                FROM solider_embeddings
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1;
            """, (vec.tolist(), vec.tolist(), threshold, vec.tolist()))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                person_name, source_camera, similarity = row
                return {
                    'person_name': person_name,
                    'similarity': float(similarity),
                    'source_camera': source_camera,
                    'match_type': 'solider_cosine'
                }, float(similarity)

            return None, 0.0

        except Exception as e:
            print(f"[ERROR] SOLIDER cosine search failed: {e}")
            return None, 0.0

    def find_transreid_match_cosine(self, query_embedding, threshold=0.7):
        """Find person match using TransReID embeddings (768-dim) with cosine similarity."""
        try:
            vec = query_embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                print("[DEBUG] TransReID: Zero norm embedding")
                return None, 0.0
            vec = vec / norm

            conn = self.db_manager.get_connection()
            if not conn:
                print("[DEBUG] TransReID: No database connection")
                return None, 0.0

            cur = conn.cursor()

            # First check if table has any data
            cur.execute("SELECT COUNT(*) FROM transreid_embeddings")
            count = cur.fetchone()[0]
            print(f"[DEBUG] TransReID: Found {count} embeddings in database")
            print(f"[DEBUG] TransReID: Query embedding dim={len(vec)}, first 5 values={vec[:5].tolist()}")

            # Get best match regardless of threshold, then filter
            try:
                cur.execute("""
                    SELECT person_name, camera_id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM transreid_embeddings
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT 1;
                """, (vec.tolist(), vec.tolist()))

                row = cur.fetchone()
                print(f"[DEBUG] TransReID: Query executed, row={row}")
            except Exception as query_error:
                print(f"[ERROR] TransReID: Query failed - {query_error}")
                cur.close()
                conn.close()
                return None, 0.0

            cur.close()
            conn.close()

            if row:
                person_name, source_camera, similarity = row
                similarity = float(similarity)

                if similarity >= threshold:
                    print(f"[DEBUG] TransReID: Match found - {person_name} with similarity {similarity:.3f}")
                    return {
                        'person_name': person_name,
                        'similarity': similarity,
                        'source_camera': source_camera,
                        'match_type': 'transreid_cosine'
                    }, similarity
                else:
                    print(f"[DEBUG] TransReID: Best match {person_name} ({similarity:.3f}) below threshold {threshold}")
                    # Return None for match but actual similarity score
                    return None, similarity

            print(f"[DEBUG] TransReID: No embeddings found in database")
            return None, 0.0

        except Exception as e:
            print(f"[ERROR] TransReID cosine search failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    def find_bpbreid_match_cosine(self, query_embeddings, query_visibility, threshold=0.7, min_visible_parts=3):
        """
        Find person match using BPBreID embeddings (6 body parts) with cosine similarity.
        Queries each body part separately and aggregates results.

        Args:
            query_embeddings: torch.Tensor [6, 512] or numpy array - 6 body parts
            query_visibility: torch.Tensor [6] or numpy array - visibility scores
            threshold: Minimum average similarity to consider a match (0.0 to 1.0)
            min_visible_parts: Minimum number of visible parts required for matching

        Returns:
            (match_dict, similarity) tuple or (None, 0.0) if no match
        """
        try:
            # Convert to numpy
            if hasattr(query_embeddings, 'cpu'):
                query_embeddings = query_embeddings.cpu().numpy()
            if hasattr(query_visibility, 'cpu'):
                query_visibility = query_visibility.cpu().numpy()

            conn = self.db_manager.get_connection()
            if not conn:
                print("[DEBUG] BPBreID: No database connection")
                return None, 0.0

            cur = conn.cursor()

            # Check if table has any data
            cur.execute("SELECT COUNT(DISTINCT face_id) FROM bpbreid_embeddings")
            count = cur.fetchone()[0]
            if count == 0:
                print("[DEBUG] BPBreID: No embeddings in database")
                cur.close()
                conn.close()
                return None, 0.0

            print(f"[DEBUG] BPBreID: Found {count} persons in database")

            # Query each visible body part and aggregate scores per person
            # Using visibility threshold of 0.5 (only match visible parts)
            part_similarities = {}  # {face_id: [similarities for each part]}

            for part_id in range(6):
                # Skip if query part is not visible
                if query_visibility[part_id] < 0.5:
                    continue

                # Normalize query embedding
                part_emb = query_embeddings[part_id].astype(np.float32)
                norm = np.linalg.norm(part_emb)
                if norm == 0:
                    continue
                part_emb = part_emb / norm

                # Query this body part from database (only visible parts)
                cur.execute("""
                    SELECT face_id,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM bpbreid_embeddings
                    WHERE body_part_id = %s
                      AND visibility >= 0.5
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT 100;
                """, (part_emb.tolist(), part_id, part_emb.tolist()))

                rows = cur.fetchall()

                # Aggregate similarities per face_id
                for face_id, similarity in rows:
                    if face_id not in part_similarities:
                        part_similarities[face_id] = []
                    part_similarities[face_id].append(float(similarity))

            cur.close()
            conn.close()

            if not part_similarities:
                print("[DEBUG] BPBreID: No matches found for any body part")
                return None, 0.0

            # Find best match: person with highest average similarity across matched parts
            best_face_id = None
            best_similarity = 0.0
            best_matched_parts = 0

            for face_id, similarities in part_similarities.items():
                num_matched_parts = len(similarities)

                # Require minimum number of matched parts
                if num_matched_parts < min_visible_parts:
                    continue

                avg_similarity = sum(similarities) / len(similarities)

                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_face_id = face_id
                    best_matched_parts = num_matched_parts

            if best_face_id is None or best_similarity < threshold:
                print(f"[DEBUG] BPBreID: Best match below threshold ({best_similarity:.3f} < {threshold})")
                return None, best_similarity

            # Get person name from face_embeddings
            conn = self.db_manager.get_connection()
            if not conn:
                return None, 0.0

            cur = conn.cursor()
            cur.execute("SELECT person_name FROM face_embeddings WHERE id = %s", (best_face_id,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                person_name = row[0]
                print(f"[DEBUG] BPBreID: Match found - {person_name} (face_id={best_face_id}) "
                      f"with similarity {best_similarity:.3f} ({best_matched_parts} parts)")

                return {
                    'person_name': person_name,
                    'face_id': best_face_id,
                    'similarity': best_similarity,
                    'matched_parts': best_matched_parts,
                    'match_type': 'bpbreid_cosine'
                }, best_similarity

            return None, 0.0

        except Exception as e:
            print(f"[ERROR] BPBreID cosine search failed: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

