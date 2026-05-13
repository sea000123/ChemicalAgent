
## windows powershell:
- 启动所有模块
```bash
conda activate mermaid
$env:JAVA_HOME="C:\Program Files\Java\jdk1.8.0_202"
$env:Path="$env:JAVA_HOME\bin;$env:Path"
java -version

#  foreground
cd ..
cd janusgraph-1.1.0
.\bin\gremlin-server.bat .\conf\gremlin-server\gremlin-server.yaml

# mermaid pipeline
conda activate mermaid
# paste api 
$env:POPPLER_PATH="C:\Program Files\poppler-25.12.0\Library\bin"
$env:Path="$env:POPPLER_PATH;$env:Path"
$env:NO_PROXY="genaiapi.shanghaitech.edu.cn"
$env:HF_ENDPOINT="https://hf-mirror.com"

visualheist --config ./scripts/startup.json
dataraider --config ./scripts/startup.json *> .\dataraider_debug.log

# 清空图
@'
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from gremlin_python.driver.serializer import GraphBinarySerializersV1
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

conn = DriverRemoteConnection(
    "ws://localhost:8182/gremlin",
    "g",
    message_serializer=GraphBinarySerializersV1()
)

g = traversal().withRemote(conn)

try:
    g.V().drop().iterate()
    print("Vertices:", g.V().count().next())
    print("Edges:", g.E().count().next())
finally:
    conn.close()
'@ | python -


kgwizard transform ./Results/JSON --output_dir ./Results/KGIntermediate --schema photo --graph_name g --address ws://localhost --port 8182 --output_file "C:/Users/user/Documents/GitHub/MERMaid/Results/Graphs/g.graphml"
# org普通有机反应，echem电化学反应，photo光化学反应

kgwizard parse ./Results/KGIntermediate `
  --schema photo `
  --graph_name g `
  --address ws://localhost `
  --port 8182

# 查看图
@'
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from gremlin_python.driver.serializer import GraphBinarySerializersV1
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

conn = DriverRemoteConnection(
    "ws://localhost:8182/gremlin",
    "g",
    message_serializer=GraphBinarySerializersV1()
)

g = traversal().withRemote(conn)

try:
    print("Vertices:", g.V().count().next())
    print("Edges:", g.E().count().next())
    print("Sample vertices:")
    print(g.V().limit(5).valueMap(True).toList())
    print("Sample edges:")
    print(g.E().limit(5).label().toList())
finally:
    conn.close()
'@ | python -

kgwizard parse ./Results/KGIntermediate `
  --schema photo `
  --graph_name g `
  --address ws://localhost `
  --port 8182 `
  --output_file "C:/Users/user/Documents/GitHub/MERMaid/Results/Graphs/g.graphml"


```
mermaid RUN   --config ./scripts/startup.json
mermaid CFG   --out_location ./Results/JSON/out.json



- test api
```bash
python -c "import os, requests, json; url='https://genaiapi.shanghaitech.edu.cn/api/v1/start'; headers={'Authorization':'Bearer '+os.environ.get('OPENAI_API_KEY',''), 'Content-Type':'application/json'}; payload={'model':'GPT-4.1-mini','messages':[{'role':'user','content':'你好'}]}; r=requests.post(url, headers=headers, json=payload, timeout=60); print(r.status_code); print(r.text[:2000])"
```

- `one_pdf.graphml`: 使用论文中作为例图的pdf进行绘制