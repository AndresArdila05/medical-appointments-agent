"""
Agente principal para el agendamiento de citas médicas
"""

from typing import Literal
from pydantic import ValidationError
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from config.settings import Config
from database.db_manager import DatabaseManager
from models.schemas import AgentState, IdentificacionPaciente, DatosCita


class MedicalAppointmentAgent:
    """Agente para manejo de citas médicas"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.google_api_key
        )
        self.app = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Construye el flujo de trabajo del agente"""
        workflow = StateGraph(AgentState)
        
        # Agregar nodos
        workflow.add_node("clasificar_intencion", self._clasificar_intencion)
        workflow.add_node("procesar_saludo", self._procesar_saludo)
        workflow.add_node("procesar_documento", self._procesar_documento)
        workflow.add_node("procesar_cita", self._procesar_cita)
        workflow.add_node("finalizar_conversacion", self._finalizar_conversacion)
        workflow.add_node("manejar_otro_tema", self._manejar_otro_tema)
        
        # Configurar el punto de entrada
        workflow.set_conditional_entry_point(
            self._route_entry,
            {
                "clasificar_intencion": "clasificar_intencion",
                "procesar_documento": "procesar_documento",
                "procesar_cita": "procesar_cita",
                "finalizar_conversacion": "finalizar_conversacion",
            }
        )
        
        # Configurar las transiciones
        workflow.add_conditional_edges(
            "clasificar_intencion",
            self._route_after_classification,
            {
                "procesar_saludo": "procesar_saludo",
                "procesar_documento": "procesar_documento",
                "manejar_otro_tema": "manejar_otro_tema",
            }
        )
        
        workflow.add_edge("procesar_saludo", END)
        workflow.add_edge("procesar_documento", END)
        workflow.add_edge("procesar_cita", END)
        workflow.add_edge("finalizar_conversacion", END)
        workflow.add_edge("manejar_otro_tema", END)
        
        return workflow.compile()
    
    def _route_entry(self, state: AgentState) -> str:
        """Determina el punto de entrada basado en el estado actual"""
        if not state.get("intent"):
            return "clasificar_intencion"
        elif state.get("document_number") and state.get("eps") and state.get("medical_specialty"):
            return "finalizar_conversacion"
        elif state.get("document_number") and not state.get("eps"):
            return "procesar_cita" # If document is present, move to collecting appointment details
        elif state.get("intent") == "agendar_cita" and not state.get("document_number"):
            return "procesar_documento"
        return "clasificar_intencion"
    
    def _route_after_classification(self, state: AgentState) -> str:
        """Ruta después de la clasificación de intención"""
        intent = state.get("intent")
        if intent == "saludo":
            return "procesar_saludo"
        elif intent == "agendar_cita":
            return "procesar_documento"
        else:
            return "manejar_otro_tema"
    
    def _clasificar_intencion(self, state: AgentState) -> AgentState:
        """Clasifica la intención del mensaje del usuario"""
        last_user_message = state["messages"][-1].content.lower()
        
        # Clasificación más flexible para saludos y afirmaciones
        saludos = ["hola", "buenos días", "buenas tardes", "buenas noches", "hi", "hey", "saludos"]
        cita_keywords = ["cita", "agendar", "reservar", "programar", "turno", "consulta"]
        # Añadir respuestas afirmativas para "sí"
        afirmativas = ["si", "sí", "claro", "por favor", "me gustaría", "afirmativo", "dale"] 
        
        intent = "otro" # Valor por defecto
        
        # Lógica para identificar intención de agendar cita incluso con una simple afirmación
        # Esto se activa si la etapa anterior fue una pregunta directa para agendar cita
        if state.get("conversation_stage") == "greeting" and any(affirmative in last_user_message for affirmative in afirmativas):
            intent = "agendar_cita"
        elif any(saludo in last_user_message for saludo in saludos):
            if any(keyword in last_user_message for keyword in cita_keywords):
                intent = "agendar_cita"
            else:
                intent = "saludo"
        elif any(keyword in last_user_message for keyword in cita_keywords):
            intent = "agendar_cita"
        else:
            # Usar LLM para casos ambiguos
            system_prompt = (
                "Analiza el siguiente mensaje del usuario. Clasifica la intención como:\n"
                "- 'saludo' si es un saludo general o conversación casual\n"
                "- 'agendar_cita' si menciona querer agendar una cita médica\n"
                "- 'otro' para cualquier otro caso\n"
                "Responde únicamente con la clasificación."
            )
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=last_user_message)
            ])
            
            llm_intent = response.content.strip().lower()
            if llm_intent in ["saludo", "agendar_cita", "otro"]:
                intent = llm_intent
        
        new_state = {**state, "intent": intent}
        self.db_manager.save_conversation(new_state)
        return new_state
    
    def _procesar_saludo(self, state: AgentState) -> AgentState:
        """Procesa un saludo del usuario"""
        ai_message = AIMessage(content=(
            "¡Hola! ¡Qué gusto saludarte! 😊 "
            "Estoy aquí para ayudarte con el agendamiento de citas médicas. "
            "¿Te gustaría agendar una cita médica hoy?"
        ))
        
        new_state = {
            **state,
            "conversation_stage": "greeting",
            "messages": state["messages"] + [ai_message]
        }
        
        self.db_manager.save_conversation(new_state)
        return new_state
    
    def _procesar_documento(self, state: AgentState) -> AgentState:
        """Procesa la recolección de datos del documento"""
        last_user_message = state["messages"][-1].content
        
        # Si no tenemos documento, necesitamos solicitarlo
        if not state.get("document_number"):
            mensaje_lower = last_user_message.lower()
            
            # Palabras clave que indican que el usuario quiere agendar pero no ha dado documento
            keywords_sin_documento = ["agendar", "cita", "quiero", "necesito"]
            
            # También considerar respuestas afirmativas simples si la etapa anterior fue "greeting"
            afirmativas_simples = ["si", "sí", "claro", "por favor", "me gustaría", "afirmativo", "dale"]

            # Comprobar si el mensaje actual es una afirmación simple después de un saludo,
            # O si contiene keywords de agendamiento sin datos de documento.
            if (state.get("conversation_stage") == "greeting" and any(affirmative in mensaje_lower for affirmative in afirmativas_simples)) or \
               (any(keyword in mensaje_lower for keyword in keywords_sin_documento) and not any(char.isdigit() for char in last_user_message)):
                
                # Solicitar documento por primera vez
                ai_message = AIMessage(content=(
                    "¡Perfecto! Para agendar tu cita médica, necesito que me proporciones "
                    "tu tipo y número de documento. Por ejemplo: 'Cédula 12345678' o 'CC 12345678'"
                ))
                new_state = {
                    **state,
                    "conversation_stage": "requesting_document",
                    "messages": state["messages"] + [ai_message],
                    "intent": "agendar_cita" # Aseguramos que la intención sea agendar_cita
                }
            else:
                # Intentar extraer documento del mensaje actual
                structured_llm_doc = self.llm.with_structured_output(IdentificacionPaciente)
                try:
                    extracted_data = structured_llm_doc.invoke([
                        SystemMessage(content=(
                            "Extrae el tipo de documento (como 'Cédula', 'CC', 'Cédula de Ciudadanía', etc.) "
                            "y el número de documento del siguiente texto. Si no encuentras esta información, "
                            "responde con 'unknown' para ambos campos."
                        )),
                        HumanMessage(content=f"Extrae tipo y número de documento de: '{last_user_message}'")
                    ])
                    
                    # Verificar si realmente se extrajo información válida
                    if (extracted_data.document_type.lower() != "unknown" and 
                        extracted_data.document_number.lower() != "unknown" and
                        any(char.isdigit() for char in extracted_data.document_number)):
                        
                        ai_message = AIMessage(content=(
                            f"¡Perfecto! He registrado tu documento: {extracted_data.document_type} "
                            f"{extracted_data.document_number}. "
                            f"Ahora necesito que me indiques tu EPS y la especialidad médica que necesitas. "
                            f"Por ejemplo: 'Sanitas, Medicina General'"
                        ))
                        
                        new_state = {
                            **state,
                            "document_type": extracted_data.document_type,
                            "document_number": extracted_data.document_number,
                            "conversation_stage": "collecting_document",
                            "messages": state["messages"] + [ai_message]
                        }
                    else:
                        # No se pudo extraer documento válido
                        ai_message = AIMessage(content=(
                            "No pude identificar tu tipo y número de documento en el mensaje. "
                            "¿Podrías proporcionármelo de nuevo, por favor? "
                            "Por ejemplo: 'Cédula 12345678' o 'CC 12345678'"
                        ))
                        new_state = {
                            **state,
                            "conversation_stage": "requesting_document", 
                            "messages": state["messages"] + [ai_message]
                        }
                        
                except (ValidationError, ValueError, Exception) as e:
                    ai_message = AIMessage(content=(
                        "No pude identificar tu tipo y número de documento en el mensaje. "
                        "¿Podrías proporcionármelo de nuevo, por favor? "
                        "Por ejemplo: 'Cédula 12345678' o 'CC 12345678'"
                    ))
                    new_state = {
                        **state,
                        "conversation_stage": "requesting_document", 
                        "messages": state["messages"] + [ai_message]
                    }
        else:
            # Ya tenemos documento, solicitar EPS y especialidad
            ai_message = AIMessage(content=(
                "Ahora necesito que me indiques tu EPS y la especialidad médica que necesitas. "
                "Por ejemplo: 'Sanitas, Medicina General'"
            ))
            new_state = {
                **state,
                "conversation_stage": "requesting_appointment_data",
                "messages": state["messages"] + [ai_message]
            }
        
        self.db_manager.save_conversation(new_state)
        return new_state
    
    def _procesar_cita(self, state: AgentState) -> AgentState:
            """Procesa la recolección de datos de la cita"""
            last_user_message = state["messages"][-1].content
            
            structured_llm_cita = self.llm.with_structured_output(DatosCita)
            try:
                extracted_data = structured_llm_cita.invoke([
                    SystemMessage(content=(
                        "Extrae la EPS (Entidad Promotora de Salud) y la especialidad médica del siguiente texto del usuario. "
                        "Presta atención a sinónimos o frases que indiquen estos datos. "
                        "Si el usuario menciona una EPS, regístrala. Si menciona una especialidad (e.g., 'dermatología', 'medicina general'), regístrala. "
                        "No te limites solo al formato 'EPS, Especialidad'. Si no encuentras una de las dos, responde con 'unknown' solo para ese campo. "
                        "Ejemplos de especialidades: Medicina General, Dermatología, Cardiología, Pediatría."
                    )),
                    HumanMessage(content=f"Texto del usuario: '{last_user_message}'")
                ])
                
                mensaje_lower = last_user_message.lower()

                # More flexible validation: check if extracted data is not 'unknown' or empty
                # and if the key elements of the extracted data appear in the user's message
                # to prevent hallucination.
                
                eps_found = (extracted_data.eps.lower() != "unknown" and 
                            extracted_data.eps.strip() != "" and
                            any(word in mensaje_lower for word in extracted_data.eps.lower().split()))

                specialty_found = (extracted_data.medical_specialty.lower() != "unknown" and
                                extracted_data.medical_specialty.strip() != "" and
                                any(word in mensaje_lower for word in extracted_data.medical_specialty.lower().split()))
                
                # If both EPS and specialty were successfully extracted
                if eps_found and specialty_found:
                    ai_message = AIMessage(content=(
                        f"¡Excelente! He registrado todos tus datos:\n"
                        f"• Documento: {state.get('document_type')} {state.get('document_number')}\n"
                        f"• EPS: {extracted_data.eps}\n"
                        f"• Especialidad: {extracted_data.medical_specialty}\n\n"
                        f"Tu solicitud de cita médica ha sido registrada correctamente. "
                        f"Pronto te contactaremos para confirmar la disponibilidad. "
                        f"¡Gracias por usar nuestro servicio!"
                    ))
                    
                    new_state = {
                        **state,
                        "eps": extracted_data.eps,
                        "medical_specialty": extracted_data.medical_specialty,
                        "conversation_stage": "completed",
                        "messages": state["messages"] + [ai_message]
                    }
                else:
                    # If either EPS or specialty (or both) are missing or invalid
                    missing_info = []
                    if not eps_found:
                        missing_info.append("tu EPS")
                    if not specialty_found:
                        missing_info.append("la especialidad médica")
                    
                    if missing_info:
                        request_message = f"Necesito que me indiques {', y '.join(missing_info)}."
                    else:
                        request_message = "No pude identificar correctamente tu EPS y especialidad médica."

                    ai_message = AIMessage(content=(
                        f"{request_message} ¿Podrías indicármelas nuevamente de forma clara? "
                        "Por ejemplo: 'Sanitas, Dermatología' o 'Sura, Medicina General'"
                    ))
                    new_state = {
                        **state,
                        "messages": state["messages"] + [ai_message]
                    }
                    
            except (ValidationError, ValueError, Exception) as e:
                # Fallback for any parsing error
                ai_message = AIMessage(content=(
                    "Hubo un problema al procesar tu solicitud de EPS y especialidad. "
                    "¿Podrías indicármelas nuevamente? "
                    "Por ejemplo: 'Sanitas, Dermatología' o 'Sura, Medicina General'"
                ))
                new_state = {
                    **state,
                    "messages": state["messages"] + [ai_message]
                }
            
            self.db_manager.save_conversation(new_state)
            return new_state
    
    def _finalizar_conversacion(self, state: AgentState) -> AgentState:
        """Finaliza la conversación"""
        self.db_manager.save_conversation(state)
        return state
    
    def _manejar_otro_tema(self, state: AgentState) -> AgentState:
        """Maneja temas fuera del alcance del agente"""
        ai_message = AIMessage(content=(
            "Entiendo tu consulta, pero mi especialidad es ayudarte con el agendamiento "
            "de citas médicas. ¿Te gustaría que te ayude a agendar una cita médica?"
        ))
        
        new_state = {
            **state,
            "messages": state["messages"] + [ai_message]
        }
        
        self.db_manager.save_conversation(new_state)
        return new_state
    
    def process_conversation(self, state: AgentState) -> AgentState:
        """Procesa una conversación completa"""
        return self.app.invoke(state)
    
    def is_conversation_complete(self, state: AgentState) -> bool:
        """Verifica si la conversación está completa"""
        return (
            state.get("document_number") is not None and
            state.get("eps") is not None and
            state.get("medical_specialty") is not None
        )