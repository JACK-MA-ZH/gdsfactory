import sys
sys.path.append('.')
from agent_r1.tool.base import BaseToolEnv, BaseImageToolEnv, BaseTool
from typing import List, Tuple, Any, Dict
import re
import json
import os
from PIL import Image
import copy
import yaml
# Mock gdsfactory for environments where it's not installed
try:
    import gdsfactory as gf
    from gdsfactory.typings import component
    from gdsfactory.generic_tech import get_generic_pdk
    
    # Initialize PDK
    gf.config.rich_output()
    PDK = get_generic_pdk()
    PDK.activate()
    GDS_INSTALLED = True

except ImportError:
    print("WARNING: gdsfactory not installed. Using mock objects for DRCToolEnv.")
    GDS_INSTALLED = False
    # Mock necessary types for type hinting
    class MockComponent:
        def to_json(self): return "{}"
        def copy(self): return self
    component = MockComponent


class DRCToolEnv(BaseImageToolEnv):
    """
    一个支持批处理的 DRC (Design Rule Check) 环境。

    这个环境内部管理一个组件列表 (批次)，并可以并行处理
    应用于每个组件的工具调用。
    """
    def __init__(self, tools: List[BaseTool], max_tool_response_length: int, data_root_dir: str = "."):
        self.tools = tools
        self.tool_map: Dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.max_tool_response_length = max_tool_response_length
        self.data_root_dir = data_root_dir

        # --- 批处理状态 ---
        self.batch_size: int = 0
        self.components: List[component] = []
        self.op_counts: List[int] = []
        # ---------------------
    
    def reset(self, components: List[component] = None, gds_paths: List[str] = None):
        """
        重置环境，加载一个批次的 GDS 文件或组件。
        """
        if gds_paths:
            self.batch_size = len(gds_paths)
            self.components = []
            for gds_path in gds_paths:
                
                abs_path = os.path.join(self.data_root_dir, gds_path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"GDS file not found at {abs_path}")
                
                c = gf.import_gds(abs_path, read_named_references=True)
                if not hasattr(c, 'named_references'):
                    c.named_references = {f"p{i}": poly for i, poly in enumerate(c.get_polygons(by_spec=False))}
                self.components.append(c)

        elif components:
            self.batch_size = len(components)
            self.components = components
        
        else:
            raise ValueError("必须提供 'components' 或 'gds_paths' 列表。")

        self.op_counts = [0] * self.batch_size

    def get_drc_violations(self) -> List[dict]:
        """为批次中的每个组件运行 DRC 检查，返回一个结果列表。"""
       
        batch_results = []
        for component in self.components:
            if component is None or not component.get_polygons():
                batch_results.append({"count": 0, "errors_text": "No polygons found.", "errors_json": [], "bboxes": [], "component": component})
                continue
            
            try:
                spacing_errors = component.drc_spacing(layer=(1, 0), spacing=0.1)
                width_errors = component.drc_width(layer=(1, 0), min_width=0.12)
                
                errors = []
                for poly in spacing_errors.get_polygons():
                    errors.append({"type": "min_spacing", "bbox": poly.bounds})
                for poly in width_errors.get_polygons():
                    errors.append({"type": "min_width", "bbox": poly.bounds})

                errors_text = "\n".join([f"ERROR: {e['type']} at {e['bbox']}" for e in errors]) if errors else "No DRC errors found."
                
                batch_results.append({
                    "count": len(errors),
                    "errors_text": errors_text,
                    "errors_json": errors,
                    "bboxes": [e['bbox'] for e in errors],
                    "component": component # 返回组件状态以供 Fix 阶段使用
                })
            except Exception as e:
                batch_results.append({"count": -1, "errors_text": f"DRC check failed: {e}", "errors_json": [], "bboxes": [], "component": component})
        
        return batch_results

    def get_schematic(self, item_index: int) -> str:
        """按索引返回单个组件的 JSON 字符串表示。"""
        if item_index >= self.batch_size:
            raise IndexError(f"索引 {item_index} 超出批次大小 {self.batch_size}")
        
        component = self.components[item_index]
        return  component.netlist() if component else "{}"

    def get_image(self, item_index: int, bbox: Tuple[float, float, float, float] = None) -> Image.Image:
        """按索引渲染单个组件的 PIL 图像。"""
        if item_index >= self.batch_size:
            raise IndexError(f"索引 {item_index} 超出批次大小 {self.batch_size}")

        component = self.components[item_index]
        
        if not component or not hasattr(gf, 'plot'):
            return Image.new('RGB', (100, 100), color='white')
        
        kwargs = {}
        if bbox:
            dx, dy = (bbox[2] - bbox[0]) * 0.5, (bbox[3] - bbox[1]) * 0.5
            kwargs['xlim'] = (bbox[0] - dx, bbox[2] + dx)
            kwargs['ylim'] = (bbox[1] - dy, bbox[3] + dy)
        
        img_array = gf.plot.get_image(component, **kwargs)
        return Image.fromarray(img_array, 'RGBA')

    def step(self, raw_responses: str) -> Tuple[str, List[Image.Image], List[bool], bool]:
       
        
        
        
        # (假定) DRCBaseTool 已导入
        # from agent_r1.tool.tools.drc_tool import DRCBaseTool

  
        raw_response = raw_responses
        component = self.components # 获取此项的组件
            
        tool_calls = self.extract_tool_calls(raw_response)
            
        if not tool_calls:
                # 此项没有工具调用，结束
            batch_formatted_responses.append("")
            batch_successes.append([])
            batch_active.append(False) # 不再活动
            continue

        tool_responses_content = []
        tool_successes = []

        for tool_call in tool_calls:
            tool_name, tool_args = tool_call.get("name"), tool_call.get("arguments", {})
            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                    
                    # [关键] 将工具应用于特定的组件实例
                result = tool.execute(args=tool_args, component=component) 
                    
                tool_responses_content=(result["content"])
                tool_successes=(result["success"])
                    
                    # 更新此项的操作计数
                if result["success"]: 
                    self.op_counts[i] += 1 
            else:
                tool_responses_content=(f"Error: Tool '{tool_name}' not found.")
                tool_successes.append(False)
            
        formatted_response = self.format_tool_response(tool_responses_content)
        batch_formatted_responses=(formatted_response)
        batch_successes=(tool_successes)
        batch_active=(True) # 此项仍在活动

        return batch_formatted_responses, batch_images, batch_successes, batch_active

    def stop(self, raw_responses: List[str]) -> List[bool]:
        """
        [批处理方法]
        检查批次中的每个响应是否应停止。
        """
        return ["<answer>" in resp or not self.extract_tool_calls(resp) for resp in raw_responses]

    # --- 辅助函数 (无需修改) ---
    def extract_tool_calls(self, raw_response: str) -> List[Dict]:
        try:
            match = re.search(r"<tool_code>(.*?)</tool_code>", raw_response, re.DOTALL)
            if match:
                tool_calls_str = match.group(1).strip()
                tool_calls = json.loads(tool_calls_str)
                return tool_calls if isinstance(tool_calls, list) else [tool_calls]
        except (json.JSONDecodeError, AttributeError):
            return []
        return []

    def format_tool_response(self, tool_responses: List[str]) -> str:
        if not tool_responses: return ""
        response_data = json.dumps({"results": [{"content": resp} for resp in tool_responses]})
        if len(response_data) > self.max_tool_response_length:
            response_data = response_data[:self.max_tool_response_length] + "..."
        return f"<tool_response>{response_data}</tool_response>"


if __name__ == '__main__':
    if GDS_INSTALLED:
        try:
            from agent_r1.tool.tools.drc_tool import MovePolygonTool
            import gdsfactory as gf

            print("--- Testing Batched DRCToolEnv ---")
            env = DRCToolEnv(tools=[MovePolygonTool()], max_tool_response_length=512)
            
            # --- 批次 1: poly1 ---
            c1 = gf.Component("env_test_1")
            c1.named_references: Dict[str, component] = {}
            c1.named_instances: Dict[str, gf.ComponentReference] = {}
            p1 = gf.Component()
            p1.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=(1, 0))
            ref1 = c1.add_ref(p1)
            c1.named_references["poly1"] = p1
            c1.named_instances["poly1"] = ref1
            
            # --- 批次 2: poly2 ---
            c2 = gf.Component("env_test_2")
            c2.named_references: Dict[str, component] = {}
            c2.named_instances: Dict[str, gf.ComponentReference] = {}
            p2 = gf.Component()
            p2.add_polygon([(10, 10), (11, 10), (11, 11), (10, 11)], layer=(1, 0))
            ref2 = c2.add_ref(p2)
            c2.named_references["poly2"] = p2
            c2.named_instances["poly2"] = ref2

            # [批处理] 重置
            env.reset(components=[c1, c2])
            assert env.batch_size == 2
            assert env.op_counts == [0, 0]

            # [批处理] LLM 响应
            llm_response_1 = """
            <tool_code>
            [{"name": "op_move_polygon", "arguments": {"polygon_name": "poly1", "dx": 5, "dy": 5}}]
            </tool_code>
            """
            llm_response_2 = """
            <tool_code>
            [{"name": "op_move_polygon", "arguments": {"polygon_name": "poly2", "dx": -2, "dy": 0}}]
            </tool_code>
            """
            
            # [批处理] Step
            batch_resp, _, batch_success, batch_active = env.step(llm_response_1)
            
            print(f"Batch responses: {batch_resp}")
            print(f"Batch success: {batch_success}")
            print(f"Batch active: {batch_active}")
            print(f"Op counts: {env.op_counts}")

            assert env.op_counts == [1, 1]
            assert batch_active == [True, True]
            assert batch_success[0][0] is True and batch_success[1][0] is True
            assert 'op_move_polygon' in batch_resp[0]
            assert 'op_move_polygon' in batch_resp[1]

            # [批处理] Stop (一个停止, 一个继续)
            llm_stop_1 = "<answer>I am done.</answer>"
            llm_stop_2 = "<tool_code>[{\"name\": \"op_move_polygon\", \"arguments\": {}}]</tool_code>" # 模拟继续
            stop_flags = env.stop([llm_stop_1, llm_stop_2])
            print(f"Stop flags: {stop_flags}")
            assert stop_flags == [True, False]

            # [批处理] get_drc_violations
            drc_results = env.get_drc_violations()
            print(f"DRC results count: {len(drc_results)}")
            assert len(drc_results) == 2
            
            # [按索引] get_schematic
            schema_1 = env.get_schematic(item_index=0)
            print(f"Schema 1: {schema_1}...")
            assert "env_test_1" in schema_1

            print("\n--- Batched DRCToolEnv Test Passed ---")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An environment test failed: {e}")
    else:
        print("Skipping DRCToolEnv test because gdsfactory is not installed.")