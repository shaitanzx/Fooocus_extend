import os
import sys
##import folder_paths

##custom_nodes_path = os.path.join(folder_paths.base_path, "custom_nodes")
custom_nodes_path = os.path.join(os.path.abspath(__file__), "custom_nodes")
##onebuttonprompt_path = os.path.join(custom_nodes_path, "OneButtonPrompt")
onebuttonprompt_path = os.path.join(os.path.abspath(__file__), "onebuttonprompt")
sys.path.append(onebuttonprompt_path)

from .OneButtonPromptNodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']