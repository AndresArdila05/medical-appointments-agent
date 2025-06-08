"""
Agente de IA para agendamiento de citas médicas
Archivo principal de ejecución
"""

import uuid
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import Config
from agents.medical_agent import MedicalAppointmentAgent
from database.db_manager import DatabaseManager
from utils.graph_visualizer import save_graph_image


def mostrar_estadisticas(db_manager: DatabaseManager):
    """Muestra estadísticas de la base de datos"""
    stats = db_manager.get_conversation_stats()
    print("\n" + "="*50)
    print("📊 ESTADÍSTICAS DE LA BASE DE DATOS")
    print("="*50)
    print(f"Total de conversaciones: {stats['total_conversations']}")
    print(f"Conversaciones completadas: {stats['completed_conversations']}")
    print(f"Tasa de completación: {stats['completion_rate']}")
    print(f"EPS únicas registradas: {stats['unique_eps']}")
    print(f"Especialidades únicas: {stats['unique_specialties']}")
    
    print("\n📋 ÚLTIMAS 5 CONVERSACIONES:")
    recent = db_manager.get_recent_conversations(5)
    for conv in recent:
        conv_id, created_at, status, doc_num, eps, specialty = conv
        print(f"  • {conv_id[:8]}... | {created_at[:16]} | {status} | Doc: {doc_num or 'N/A'} | EPS: {eps or 'N/A'}")
    print("="*50)


def main():
    """Función principal del agente"""
    try:
        # Inicializar componentes
        config = Config()
        db_manager = DatabaseManager()
        agent = MedicalAppointmentAgent(config, db_manager)
        
        # Generar imagen del grafo
        save_graph_image(agent.app, "medical_agent_graph.png")
        print("📊 Imagen del grafo guardada como 'medical_agent_graph.png'")
        
        # Inicializar conversación
        conversation_id = str(uuid.uuid4())
        initial_greeting = (
            "¡Hola! Soy tu asistente virtual para el agendamiento de citas médicas. "
            "Mi función principal es ayudarte a agendar tu cita médica. "
            "¿Cómo estás hoy?"
        )
        
        current_state = {
            "conversation_id": conversation_id,
            "messages": [AIMessage(content=initial_greeting)]
        }
        
        print(f"Agente: {initial_greeting}")
        print("---")
        print("💡 Comandos: 'stats' para estadísticas, 'salir' para terminar, 'grafo' para ver el grafo")
        print("---")

        while True:
            try:
                user_input = input("Cliente: ").strip()
                
                if user_input.lower() in ["salir", "exit", "quit"]:
                    print("¡Gracias por usar nuestro servicio! ¡Que tengas un buen día!")
                    mostrar_estadisticas(db_manager)
                    break
                
                if user_input.lower() == "stats":
                    mostrar_estadisticas(db_manager)
                    continue
                
                if user_input.lower() == "grafo":
                    print("📊 El grafo se ha guardado como 'medical_agent_graph.png'")
                    save_graph_image(agent.app, "medical_agent_graph.png")
                    continue
                
                if not user_input:
                    print("Por favor, escribe algo para continuar.")
                    continue
                
                # Procesar mensaje del usuario
                current_state["messages"].append(HumanMessage(content=user_input))
                
                # Invocar el agente
                final_state = agent.process_conversation(current_state)
                
                # Mostrar respuesta del agente
                if len(final_state["messages"]) > len(current_state["messages"]):
                    agent_response = final_state["messages"][-1].content
                    print(f"Agente: {agent_response}")
                    print("---")
                
                current_state = final_state
                
                # Verificar si la conversación ha terminado
                if agent.is_conversation_complete(final_state):
                    print("\n📝 ¡Cita registrada exitosamente!")
                    print("Estado Final Recolectado:")
                    print(f"  - ID Conversación: {final_state.get('conversation_id')}")
                    print(f"  - Documento: {final_state.get('document_type')} {final_state.get('document_number')}")
                    print(f"  - EPS: {final_state.get('eps')}")
                    print(f"  - Especialidad: {final_state.get('medical_specialty')}")
                    mostrar_estadisticas(db_manager)
                    break
                    
            except KeyboardInterrupt:
                print("\n\nConversación interrumpida por el usuario.")
                mostrar_estadisticas(db_manager)
                break
            except Exception as e:
                print(f"Ocurrió un error: {e}")
                print("Por favor, intenta de nuevo.")
                continue
                
    except Exception as e:
        print(f"Error al inicializar el sistema: {e}")
        return


if __name__ == "__main__":
    main()