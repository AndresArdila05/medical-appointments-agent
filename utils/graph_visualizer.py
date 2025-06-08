"""
Utilidad para visualizar el grafo de LangGraph
"""

import io
from typing import Optional
from PIL import Image, ImageDraw, ImageFont


def save_graph_image(app, filename: str = "graph.png"):
    """
    Guarda una imagen del grafo de LangGraph
    """
    try:
        # Intentar usar el método get_graph de LangGraph
        if hasattr(app, 'get_graph'):
            graph = app.get_graph()
            
            # Usar la funcionalidad nativa de visualización si está disponible
            if hasattr(graph, 'draw_mermaid_png'):
                graph_image_data = graph.draw_mermaid_png()
                with open(filename, 'wb') as f:
                    f.write(graph_image_data)
                print(f"✅ Grafo guardado como {filename}")
                return True
        
        # Fallback: crear una representación visual simple
        _create_simple_graph_image(filename)
        print(f"✅ Grafo simplificado guardado como {filename}")
        return True
        
    except Exception as e:
        print(f"⚠️ No se pudo generar la imagen del grafo: {e}")
        _create_simple_graph_image(filename)
        print(f"✅ Grafo simplificado guardado como {filename}")
        return False


def _create_simple_graph_image(filename: str):
    """
    Crea una representación visual simple del flujo del agente
    """
    # Crear imagen
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Colores
    node_color = '#4A90E2'