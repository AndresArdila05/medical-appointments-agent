"""
Modelos de datos para el sistema de agendamiento de citas médicas
"""

from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class IdentificacionPaciente(BaseModel):
    """Modelo para la identificación del paciente"""
    document_type: str = Field(
        description="Tipo de documento, ej: Cédula de Ciudadanía, CC, C.C."
    )
    document_number: str = Field(
        description="El número del documento"
    )


class DatosCita(BaseModel):
    """Modelo para los datos de la cita médica"""
    eps: str = Field(
        description="La EPS a la que pertenece el paciente, ej: Sanitas, Sura"
    )
    medical_specialty: str = Field(
        description="La especialidad médica requerida, ej: Dermatología, Medicina General"
    )


class AgentState(TypedDict):
    """Estado del agente de conversación"""
    conversation_id: str | None
    intent: Literal["agendar_cita", "saludo", "otro"] | None
    document_type: str | None
    document_number: str | None
    eps: str | None
    medical_specialty: str | None
    messages: List[BaseMessage]
    conversation_stage: Literal["greeting", "collecting_document", "collecting_cita", "completed"] | None