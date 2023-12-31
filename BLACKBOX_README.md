 This is a simple chat application that allows two users to communicate with each other in real-time.

### Step 1: HTML

The HTML code creates a simple user interface for the chat application. It includes an input field for the user to enter their message, a button to send the message, and a div to display the messages.

```html
<!DOCTYPE html>
<html>
 <head>
  <title>Chat P2P</title>
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.4.1/socket.io.min.js"></script>
 </head>
 <body>
  <input type="text" id="message" />
  <button onclick="sendMessage()">Enviar</button>
  <div id="messages"></div>
  <script>
  var socket = io();

  socket.on("connect", function() {
    console.log("Conectado ao servidor");
  });

  socket.on("message", function(message) {
    $("#messages").append("<p>" + message + "</p>");
  });

  function sendMessage() {
    var message = $("#message").val();
    socket.emit("message", message);
    $("#message").val("");
  }
  </script>
 </body>
</html>
```

### Step 2: JavaScript

The JavaScript code handles the real-time communication between the two users. It uses the Socket.IO library to establish a WebSocket connection between the client and the server.

When the user clicks the "Enviar" button, the sendMessage() function is called. This function gets the message from the input field and emits it to the server using the socket.emit() function.

The server then broadcasts the message to all connected clients, including the other user. When the other user receives the message, the socket.on("message") event is triggered and the message is displayed in the chat window.

```javascript
var socket = io();

socket.on("connect", function() {
  console.log("Conectado ao servidor");
});

socket.on("message", function(message

Generated by [BlackboxAI](https://www.useblackbox.ai)
