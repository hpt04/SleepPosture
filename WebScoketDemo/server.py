import asyncio
import websockets
import json
import re


def load_data(path):
    # 从文件中加载多个JSON对象
    with open(path, 'r') as file:
        content = file.read()
        # 使用正则表达式匹配每个JSON 对象
        pattern = r'\{.*?\}'
        matches = re.findall(pattern, content, re.DOTALL)
        data = [json.loads(match) for match in matches]
        return data

data = load_data("json.txt")


async def send_message(websocket):
    index = 0
    while True:
        if index < len(data):
            await websocket.send(json.dumps(data[index], indent=4))
            # print(json.dumps(data[index], indent=4))
            index += 1
            await asyncio.sleep(1.5)
        else:
            index = 0
        

async def receive_message(websocket):
    async for message in websocket:
        data = json.loads(message)
        airbag = [int(item.strip()) for item in data.get("airbag", "[]").strip('[]').split(',') if item.strip()]
        t = int(data.get("time", ""))
        print("receive message:",airbag,t)



async def websocket_handler(websocket, path):
    # 创建两个任务，分别用于发送 ping 和接收消息
    send_task = asyncio.create_task(send_message(websocket))
    receive_task = asyncio.create_task(receive_message(websocket))

    # 等待任务完成（实际中这个连接会一直保持）
    await asyncio.wait([send_task, receive_task], return_when=asyncio.FIRST_COMPLETED)

async def start_websocket_server():
    async with websockets.serve(websocket_handler, "127.0.0.1", 8000):
        print("web socket sever connected")
        await asyncio.Future()  # 运行直到被取消

loop = asyncio.get_event_loop()
websocket_server = loop.create_task(start_websocket_server())  # 创建WebSocket服务器任务
loop.run_forever()

