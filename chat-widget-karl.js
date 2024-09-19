// chat-widget.js

(function() {
    // Enhanced CSS Styles
    var styles = `
        #chat-icon {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background-color: #4a90e2;
            border-radius: 50%;
            cursor: pointer;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        #chat-icon:hover {
            background-color: #357abd;
            transform: scale(1.1);
        }
        #chat-window {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            background-color: #f5f8fa;
            border: none;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
            display: none;
            flex-direction: column;
            overflow: hidden;
            z-index: 1000;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #chat-header {
            background-color: #4a90e2;
            color: #ffffff;
            padding: 15px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            position: relative;
        }
        #chat-close {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            cursor: pointer;
            font-size: 24px;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }
        #chat-close:hover {
            opacity: 1;
        }
        #chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: #ffffff;
        }
        #chat-input {
            width: 100%;
            border: none;
            border-top: 1px solid #e1e8ed;
            padding: 15px;
            outline: none;
            font-size: 16px;
            background-color: #f5f8fa;
        }
        .user-message, .bot-message {
            margin: 10px 0;
            clear: both;
        }
        .user-message {
            float: right;
        }
        .bot-message {
            float: left;
        }
        .message {
            display: inline-block;
            padding: 10px 15px;
            border-radius: 20px;
            max-width: 80%;
            word-wrap: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .user-message .message {
            background-color: #4a90e2;
            color: #ffffff;
        }
        .bot-message .message {
            background-color: #e1e8ed;
            color: #14171a;
        }
        .thinking {
            font-style: italic;
            color: #657786;
            padding: 10px 15px;
            background-color: #f1f3f5;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 10px;
            min-width: 80px; /* Ensure minimum width to fit "Thinking..." */
            text-align: center; /* Center the text */
        }
        .error {
            color: #e0245e;
            font-style: italic;
        }
        /* Markdown Styles */
        .bot-message .message p {
            margin: 0 0 10px 0;
        }
        .bot-message .message ul, .bot-message .message ol {
            margin: 0 0 10px 20px;
            padding: 0;
        }
        .bot-message .message code {
            background-color: #f1f3f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        .bot-message .message pre {
            background-color: #f1f3f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .bot-message .message a {
            color: #1da1f2;
            text-decoration: none;
        }
        .bot-message .message a:hover {
            text-decoration: underline;
        }
    `;

    // Add styles to the document
    var styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = styles;
    document.head.appendChild(styleSheet);

    // Load Marked.js for Markdown support
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    document.head.appendChild(script);

    // Create chat icon
    var chatIcon = document.createElement('div');
    chatIcon.id = 'chat-icon';
    chatIcon.innerHTML = '&#128172;'; // Speech balloon emoji
    document.body.appendChild(chatIcon);

    // Create chat window
    var chatWindow = document.createElement('div');
    chatWindow.id = 'chat-window';
    document.body.appendChild(chatWindow);

    // Chat header
    var chatHeader = document.createElement('div');
    chatHeader.id = 'chat-header';
    chatHeader.innerText = 'Chat with Us';
    
    // Close button
    var chatClose = document.createElement('span');
    chatClose.id = 'chat-close';
    chatClose.innerHTML = '&times;'; // Close (×) symbol
    chatHeader.appendChild(chatClose);
    
    chatWindow.appendChild(chatHeader);

    // Chat messages container
    var chatMessages = document.createElement('div');
    chatMessages.id = 'chat-messages';
    chatWindow.appendChild(chatMessages);

    // Chat input
    var chatInput = document.createElement('input');
    chatInput.type = 'text';
    chatInput.id = 'chat-input';
    chatInput.placeholder = 'Type your message...';
    chatWindow.appendChild(chatInput);

    // Add starter message
    var mentorName = 'Karl'; // Replace with the actual mentor name
    var starterMessage = `Hey, this is ${mentorName}. This is my AI version trained on all my knowledge. Ask me anything...`;

    // Toggle chat window
    chatIcon.addEventListener('click', function() {
        if (chatWindow.style.display === 'none' || chatWindow.style.display === '') {
            chatWindow.style.display = 'flex';
            if (chatMessages.children.length === 0) {
                appendMessage('Bot', starterMessage, 'bot-message');
            }
        } else {
            chatWindow.style.display = 'none';
        }
    });

    // Close chat window
    chatClose.addEventListener('click', function(event) {
        event.stopPropagation(); // Prevent the click from triggering the chatIcon click event
        chatWindow.style.display = 'none';
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

        var contentElement = document.createElement('div');
        contentElement.className = 'message';

        if (sender === 'Bot' && message === 'Thinking...') {
            contentElement.className = 'thinking';
            contentElement.textContent = message;
        } else if (sender === 'Bot') {
            contentElement.innerHTML = marked.parse(message);
        } else {
            contentElement.textContent = message;
        }

        messageElement.appendChild(contentElement);
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return messageElement.id = 'msg-' + Date.now();
    }

    function removeMessage(messageId) {
        var messageElement = document.getElementById(messageId);
        if (messageElement) {
            messageElement.remove();
        }
    }
})();
