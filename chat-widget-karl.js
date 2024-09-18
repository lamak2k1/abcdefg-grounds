<!-- Chat Widget Script -->
<div id="chatbot">
  <div id="chatbot-header" onclick="toggleChatbot()">Chat with Us</div>
  <div id="chatbot-body">
    <div id="chatbot-messages"></div>
    <input type="text" id="chatbot-input" placeholder="Type your message..." onkeypress="handleKeyPress(event)" />
  </div>
</div>

<style>
  /* Chatbot Styles */
  #chatbot {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 300px;
    font-family: Arial, sans-serif;
  }

  #chatbot-header {
    background-color: #007bff;
    color: #ffffff;
    padding: 10px;
    border-radius: 10px 10px 0 0;
    cursor: pointer;
    text-align: center;
  }

  #chatbot-body {
    display: none;
    background-color: #f1f1f1;
    border: 1px solid #007bff;
    border-top: none;
    border-radius: 0 0 10px 10px;
    overflow: hidden;
  }

  #chatbot-messages {
    height: 200px;
    padding: 10px;
    overflow-y: auto;
    background-color: #ffffff;
  }

  #chatbot-messages p {
    margin: 5px 0;
  }

  #chatbot-messages .user-message {
    text-align: right;
    color: #000000;
  }

  #chatbot-messages .bot-message {
    text-align: left;
    color: #007bff;
  }

  #chatbot-input {
    width: calc(100% - 20px);
    padding: 10px;
    border: none;
    border-top: 1px solid #ccc;
    outline: none;
  }
</style>

<script>
  // Chatbot JavaScript
  function toggleChatbot() {
    const chatbotBody = document.getElementById('chatbot-body');
    chatbotBody.style.display = chatbotBody.style.display === 'none' ? 'block' : 'none';
  }

  function handleKeyPress(event) {
    if (event.key === 'Enter') {
      sendMessage();
    }
  }

  function sendMessage() {
    const inputField = document.getElementById('chatbot-input');
    const message = inputField.value.trim();
    if (message === '') return;

    appendMessage('You', message, 'user-message');
    inputField.value = '';

    // Encode the mentor name and message to use in query parameters
    const mentorName = encodeURIComponent('karl'); // Replace with your mentor's name
    const userPrompt = encodeURIComponent(message);

    // Build the URL with query parameters
    const apiUrl = `https://abcdefg-fastapi.onrender.com/chat?name=${mentorName}&prompt=${userPrompt}`;

    // Send the message to the backend API
    fetch(apiUrl, {
      method: 'POST',
    })
      .then((response) => response.json())
      .then((data) => {
        const reply = data.answer || 'Sorry, I did not understand that.';
        appendMessage('Bot', reply, 'bot-message');
      })
      .catch((error) => {
        console.error('Error:', error);
        appendMessage('Bot', 'Sorry, there was an error processing your request.', 'bot-message');
      });
  }

  function appendMessage(sender, message, className) {
    const messagesContainer = document.getElementById('chatbot-messages');
    const messageElement = document.createElement('p');
    messageElement.className = className;
    messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
</script>
