from aiohttp import web
import asyncio
import threading
import time
HTML = """
  <head>
    <title>Gardengoat</title>
    <style type="text/css">
      .navbutton{
        height:30%;
        width:33%;
        font-size:100px
      }
    .widebutton{
        height:10%;
        width:49%;
        font-size:100px
      }
    </style>
  </head>
<button class="navbutton" onclick="sendkey('forwardleft')">↖</button>
<button class="navbutton" onclick="sendkey('forward')">↑</button>
<button class="navbutton" onclick="sendkey('forwardright')">↗</button>
<button class="navbutton" onclick="sendkey('left')">←</button>
<button class="navbutton" onclick="sendkey('stop')">STOP</button>
<button class="navbutton" onclick="sendkey('right')">→</button>
<button class="navbutton" onclick="sendkey('backandturn')">↪️</button>
<button class="navbutton" onclick="sendkey('backward')">↓</button>
<button class="navbutton" onclick="sendkey('AI')">💾</button>

<button class="widebutton" onclick="sendkey('cut')">cut</button>
<button class="widebutton" onclick="sendkey('speed')">speed</button>

<script>
  function sendkey(key) {
      fetch("/".concat(key))
  }
</script>
"""

class Webagent:
    def __init__(self):
        self.key = "stop"
        self.cut = 0
        self.speed = 1
        t = threading.Thread(target=self.run_server, args=(self.aiohttp_server(),))
        t.start()
        return None

    async def index(self, request: web.Request) -> web.Response:
        return web.Response(text=HTML, content_type="text/html")

    async def key_request(self, request: web.Request) -> web.Response:
        key = request.match_info.get('key')#.encode()
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
        site = web.TCPSite(runner, port= 80)
        loop.run_until_complete(site.start())
        loop.run_forever()

    def __call__(self, *args, **kwargs):
        
        if self.key == "stop":
            action = {'left' : 0, 'right' : 0, "cut" : 0}
        elif self.key == "forward":
            action = {'left' : 1, 'right' : 1}
        elif self.key == "backward":
            action = {'left' : -1, 'right' : -1}
        elif self.key == "left":
            action = {'left' : -1, 'right' : 1}

        elif self.key == "right":
            action = {'left' : 1, 'right' : -1}

        elif self.key == "forwardleft":
            action = {'left' : 0.5, 'right' : 1}

        elif self.key == "forwardright":
            action = {'left' : 1, 'right' : 0.5}

        else:
            action = {}

        action['action'] = self.key
        
        # CUTTING:
        if self.key == "cut":
            self.cut = 1-self.cut
            action['cut'] = self.cut
            self.key=None
        return action

if __name__ == "__main__":
    import time
    server = Webagent()
    while True:
        print(server.key, server())
        time.sleep(5)
