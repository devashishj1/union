"use client"
import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send } from 'lucide-react';
import { API_URL } from '@/lib/utils';

const Home = () => {
  const [message, setMessage] = useState('');
  const [conversations, setConversations] = useState([]);
  const [assistants, setAssistants] = useState([]);
  const [selectedAssistant, setSelectedAssistant] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAssistants();
  }, []);

  const fetchAssistants = async () => {
    try {
      const response = await fetch(`${API_URL}/assistants/`);
      const data = await response.json();
      setAssistants(data.assistants);
    } catch (error) {
      console.error('Error fetching assistants:', error);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim() || !selectedAssistant) return;

    setLoading(true);
    const newMessage = { 
      role: 'user', 
      content: message
    };
    setConversations(prev => [...prev, newMessage]);

    try {
      const response = await fetch(`${API_URL}/chat/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'assistant-id': selectedAssistant,
        },
        body: JSON.stringify({
          user_id: 'default',
          message: message,
        }),
      });

      const data = await response.json();
      let content = data.response;

      try {
        // Parse the response as JSON
        const parsed = JSON.parse(content);
        console.log(parsed);
        
        content = (
          <div className="space-y-6">
            {/* Final Answer */}
            <div className="p-4 border rounded-md bg-gray-50">
              <p>{parsed?.analysis?.final_answer || "Not provided"}</p>
            </div>

            {/* Analysis Details Accordion */}
            <details className="border rounded-md p-4">
              <summary className="cursor-pointer font-medium">Show Analysis Details</summary>
              <div className="mt-4">
                <h3 className="font-semibold">Risk Assessment</h3>
                <p>{parsed.analysis?.analysis?.risk_assessment || "N/A"}</p>
              </div>
              <div className="mt-4">
                <h3 className="font-semibold">Documentation & Approvals</h3>
                <p>{parsed.analysis?.analysis?.documentation_and_approvals || "N/A"}</p>
              </div>
              <div className="mt-4">
                <h3 className="font-semibold">Procurement Strategy</h3>
                <p>{parsed.analysis?.analysis?.procurement_strategy || "N/A"}</p>
              </div>
            </details>
          </div>
        );
      } catch (error) {
        // If not valid JSON, show as plain text.
        content = <p>{content}</p>;
      }
      
      setConversations(prev => [...prev, {
        role: 'assistant',
        content: content
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
      setMessage('');
    }
  };

  return (
    <div className="max-w-3xl mx-auto h-screen bg-white flex flex-col">
      {/* Chat Container */}
      <Card className="flex-1 flex flex-col">
        {/* Assistant Selection */}
        <div className="p-4 border-b">
          <Select 
            value={selectedAssistant} 
            onValueChange={setSelectedAssistant}
          >
            <SelectTrigger className="w-full bg-white">
              <SelectValue placeholder="Select an assistant" />
            </SelectTrigger>
            <SelectContent>
              {assistants.map((assistant) => (
                <SelectItem key={assistant.id} value={assistant.id}>
                  {assistant.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Messages Area */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {conversations.map((msg, index) => (
              <div
                key={index}
                className={`flex ${
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                <div
                  className={`px-4 py-2 rounded-lg max-w-[80%] ${
                    msg.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Message Input */}
        <div className="p-4 border-t">
          <form onSubmit={sendMessage} className="flex space-x-2">
            <Textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type your message..."
              disabled={loading || !selectedAssistant}
              className="flex-1 resize-none"
              rows={1}
            />
            <Button 
              type="submit" 
              disabled={loading || !selectedAssistant}
              className="bg-blue-500 hover:bg-blue-600"
            >
              <Send className="w-4 h-4" />
            </Button>
          </form>
        </div>
      </Card>
    </div>
  );
};

export default Home;