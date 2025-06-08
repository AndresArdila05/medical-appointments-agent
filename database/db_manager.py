"""
Gestor de base de datos para el sistema de agendamiento de citas médicas
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Any
from langchain_core.messages import HumanMessage
from models.schemas import AgentState


class DatabaseManager:
    """Gestor de la base de datos SQLite"""
    
    def __init__(self, db_path: str = "medical_appointments.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa la base de datos con las tablas necesarias"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla de conversaciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    status TEXT DEFAULT 'active',
                    intent TEXT,
                    document_type TEXT,
                    document_number TEXT,
                    eps TEXT,
                    medical_specialty TEXT,
                    conversation_stage TEXT
                )
            ''')
            
            # Tabla de mensajes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            ''')
            
            # Índices para mejorar performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_conv_id ON messages(conversation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)')
            
            conn.commit()
            print("✅ Base de datos inicializada correctamente")
    
    def save_conversation(self, state: AgentState) -> bool:
        """Guarda o actualiza una conversación completa"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Determinar el estado de la conversación
                status = 'completed' if state.get("eps") and state.get("medical_specialty") else 'active'
                
                # Insertar o actualizar conversación
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, created_at, updated_at, intent, document_type, 
                     document_number, eps, medical_specialty, status, conversation_stage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    state.get("conversation_id"),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    state.get("intent"),
                    state.get("document_type"),
                    state.get("document_number"),
                    state.get("eps"),
                    state.get("medical_specialty"),
                    status,
                    state.get("conversation_stage")
                ))
                
                # Limpiar mensajes anteriores para esta conversación
                cursor.execute('DELETE FROM messages WHERE conversation_id = ?', 
                             (state.get("conversation_id"),))
                
                # Insertar todos los mensajes
                for msg in state.get("messages", []):
                    message_type = "human" if isinstance(msg, HumanMessage) else "ai"
                    cursor.execute('''
                        INSERT INTO messages (conversation_id, message_type, content, timestamp)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        state.get("conversation_id"),
                        message_type,
                        msg.content,
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"❌ Error al guardar en base de datos: {e}")
            return False
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de las conversaciones"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Estadísticas generales
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM conversations WHERE status = "completed"')
            completed_conversations = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT eps) FROM conversations WHERE eps IS NOT NULL')
            unique_eps = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT medical_specialty) FROM conversations WHERE medical_specialty IS NOT NULL')
            unique_specialties = cursor.fetchone()[0]
            
            return {
                "total_conversations": total_conversations,
                "completed_conversations": completed_conversations,
                "completion_rate": f"{(completed_conversations/total_conversations*100):.1f}%" if total_conversations > 0 else "0%",
                "unique_eps": unique_eps,
                "unique_specialties": unique_specialties
            }
    
    def get_recent_conversations(self, limit: int = 10) -> List[Tuple]:
        """Obtiene las conversaciones más recientes"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT conversation_id, created_at, status, document_number, eps, medical_specialty
                FROM conversations 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()
    
    def get_conversation_by_id(self, conversation_id: str) -> Dict[str, Any] | None:
        """Obtiene una conversación específica por ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM conversations 
                WHERE conversation_id = ?
            ''', (conversation_id,))
            
            result = cursor.fetchone()
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None