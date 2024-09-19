// chat-widget.js

document.addEventListener('DOMContentLoaded', function() {
  // Create chat icon
  var chatIcon = document.createElement('div');
  chatIcon.id = 'chat-icon';
  chatIcon.style.position = 'fixed';
  chatIcon.style.bottom = '20px';
  chatIcon.style.right = '20px';
  chatIcon.style.width = '60px';
  chatIcon.style.height = '60px';
  chatIcon.style.backgroundColor = '#007bff';
  chatIcon.style.borderRadius = '50%';
  chatIcon.style.cursor = 'pointer';
  chatIcon.style.zIndex = '1000';
  chatIcon.style.display = 'flex';
  chatIcon.style.alignItems = 'center';
  chatIcon.style.justifyContent = 'center';
  chatIcon.style.color = '#ffffff';
  chatIcon.style.fontSize = '30px';
  chatIcon.innerHTML = '&#128172;'; // Speech balloon emoji
  document.body.appendChild(chatIcon);

  // Create chat window
  var chatWindow = document.createElement('div');
  chatWindow.id = 'chat-window';
  chatWindow.style.position = 'fixed';
  chatWindow.style.bottom = '90px';
  chatWindow.style.right = '20px';
  chatWindow.style.width = '300px';
  chatWindow.style.height = '400px';
  chatWindow.style.backgroundColor = '#ffffff';
  chatWindow.style.border = '1px solid #ccc';
  chatWindow.style.borderRadius = '10px';
  chatWindow.style.boxShadow = '0 0 10px rgba(0,0,0,0.2)';
  chatWindow.style.display = 'none';
  chatWindow.style.flexDirection = 'column';
  chatWindow.style.overflow = 'hidden';
  chatWindow.style.zIndex = '1000';
  document.body.appendChild(chatWindow);

  // Chat header
  var chatHeader = document.createElement('div');
  chatHeader.style.backgroundColor = '#007bff';
  chatHeader.style.color = '#ffffff';
  chatHeader.style.padding = '10px';
  chatHeader.style.textAlign = 'center';
  chatHeader.innerText = 'Chat with Us';
  chatWindow.appendChild(chatHeader);

  // Chat messages
  var chatMessages = document.createElement('div');
  chatMessages.id = 'chat-messages';
  chatMessages.style.flex = '1';
  chatMessages.style.padding = '10px';
  chatMessages.style.overflowY = 'auto';
  chatWindow.appendChild(chatMessages);

  // Chat input
  var chatInput = document.createElement('input');
  chatInput.type = 'text';
  chatInput.id = 'chat-input';
  chatInput.placeholder = 'Type your message...';
  chatInput.style.width = '100%';
  chatInput.style.border = 'none';
  chatInput.style.borderTop = '1px solid #ccc';
  chatInput.style.padding = '10px';
  chatInput.style.outline = 'none';
  chatWindow.appendChild(chatInput);

  // Toggle chat window
  chatIcon.addEventListener('click', function() {
    chatWindow.style.display = chatWindow.style.display === 'none' ? 'flex' : 'none';
  });

  // Handle message sending
  chatInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      var message = chatInput.value.trim();
      if (message === '') return;
      appendMessage('You', message, 'user-message');
      chatInput.value = '';

      // Show "Thinking..." message
      var thinkingMessageId = appendMessage('Bot', 'Thinking...', 'bot-message thinking');

      // Encode the mentor name and message to use in query parameters
      const mentorName = encodeURIComponent('karl');
      const userPrompt = encodeURIComponent(message);

      // Send the message to your backend API
      fetch(`https://abcdefg-fastapi.onrender.com/chat?name=${mentorName}&prompt=${userPrompt}`, {
        method: 'POST',
      })
        .then((response) => response.json())
        .then((data) => {
          var reply = data.answer || 'Sorry, I did not understand that.';
          // Remove "Thinking..." message and append the actual reply
          removeMessage(thinkingMessageId);
          appendMessage('Bot', reply, 'bot-message');
        })
        .catch((error) => {
          console.error('Error:', error);
          // Remove "Thinking..." message and append the error message
          removeMessage(thinkingMessageId);
          appendMessage('Bot', 'Sorry, there was an error processing your request.', 'bot-message error');
        });
    }
  });

  function appendMessage(sender, message, className) {
    var messagesContainer = document.getElementById('chat-messages');
    var messageElement = document.createElement('div');
    messageElement.className = className;
    messageElement.style.margin = '5px 0';
    messageElement.innerHTML = '<strong>' + sender + ':</strong> ' + message;
    messagesContainer.appendChild(messageElement);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return messageElement.id = 'msg-' + Date.now(); // Return an ID for the message
  }

  function removeMessage(messageId) {
    var messageElement = document.getElementById(messageId);
    if (messageElement) {
      messageElement.remove();
    }
  }
});
