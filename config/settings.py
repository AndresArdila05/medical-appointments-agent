"""
Configuración del sistema de agendamiento de citas médicas
"""

import os
from dotenv import load_dotenv


class Config:
    """Clase de configuración del sistema"""
    
    def __init__(self):
        load_dotenv()
        self._validate_environment()
    
    @property
    def google_api_key(self) -> str:
        """Clave API de Google"""
        return os.getenv('GOOGLE_API_KEY')
    
    @property
    def model_name(self) -> str:
        """Nombre del modelo a utilizar"""
        return os.getenv('MODEL_NAME', 'gemini-2.0-flash')
    
    @property
    def temperature(self) -> float:
        """Temperatura del modelo"""
        return float(os.getenv('TEMPERATURE', '0'))
    
    @property
    def database_path(self) -> str:
        """Ruta de la base de datos"""
        return os.getenv('DATABASE_PATH', 'medical_appointments.db')
    
    def _validate_environment(self):
        """Valida que las variables de entorno necesarias estén configuradas"""
        if not self.google_api_key:
            raise ValueError(
                "No se encontró la GOOGLE_API_KEY. "
                "Asegúrate de crear un archivo .env con tu clave."
            )