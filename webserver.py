from aiohttp import web
import asyncio
import threading
import time
HTML = """
  <head>
    <title>Embedded Style Sample</title>
    <style type="text/css">
      button{
        height:33%;
        width:33%;
        font-size:100px
      }
    </style>
  </head>
<button onclick="sendkey('forwardleft')">↖</button>
<button onclick="sendkey('forward')">↑</button>
<button onclick="sendkey('forwardright')">↗</button>
<button onclick="sendkey('left')">←</button>
<button onclick="sendkey('backward')">↓</button>
<button onclick="sendkey('right')">→</button>
<button onclick="sendkey('stop')">STOP</button>
<button onclick="sendkey('stop')">STOP</button>
<button onclick="sendkey('cut')">cut</button>
<script>
  function sendkey(key) {
      fetch("/".concat(key))
  }
</script>
"""

class Webagent:
    def __init__(self):
        self.key = None
        t = threading.Thread(target=self.run_server, args=(self.aiohttp_server(),))
        t.start()
        return None

    async def index(self, request: web.Request) -> web.Response:
        return web.Response(text=HTML, content_type="text/html")

    async def key_request(self, request: web.Request) -> web.Response:
        key = request.match_info.get('key').encode()
        self.key = key
        print(self.key)
        return await self.index(request)

    def setup_routes(self, app):
        app.router.add_get('/', self.index)
        app.router.add_get('/{key}', self.key_request)

    def aiohttp_server(self):
        app = web.Application()
        self.setup_routes(app)
        runner = web.AppRunner(app)
        #web.run_app(app, port=9999)
        return runner

    def run_server(self,runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, 'localhost', 9999)
        loop.run_until_complete(site.start())
        loop.run_forever()

    def __call__(self, *args, **kwargs):
        action = {
            'left' : 0,
            'right' : 0,
            'cut' : 0
        }
        key = self.key
        if key == "stop":
            action['stop'] = True
        elif key == "forward":
            action['left'] = 1
            action['right'] = 1
        elif key == "backward":
            action['left'] = -1
            action['right'] = -1
        elif key == "left":
            action['left'] = -1
            action['right'] = 1
        elif key == "right":
            action['left'] = 1
            action['right'] = -1
        elif key == "forwardleft":
            action['left'] = 0.5
            action['right'] = 1
        elif key == "forwardright":
            action['left'] = 1
            action['right'] = 0.5
        if key == "cut":
            action['cut'] = True
        return action

if __name__ == "__main__":
    import time
    server = Webagent()
    while True:
        print(server.key)
        time.sleep(0.5)
