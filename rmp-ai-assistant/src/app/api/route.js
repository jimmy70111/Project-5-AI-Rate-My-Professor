import { NextResponse } from 'next/server';

const sysPrompt = "You are a chatbot designed to answer questions specifically about the Olympics. Remind the user of your purpose if they stray from the topic of Olympics and anything related to the Olympics.";

export async function POST(req) {
  const data = await req.json();

  const completion = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.LLAMA_API_KEY}`, // Use the API key from environment variables
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      "model": "meta-llama/llama-3.1-8b-instruct:free",
      "messages": [
        {"role": "system", "content": sysPrompt},
        ...data
      ],
      "top_p": 1,
      "temperature": 1,
      "repetition_penalty": 1,
      //"stream": true
    })
  });

  const apiResponseData = await completion.json();

  // Assuming the API returns a 'choices' array with a 'message' field
  if (apiResponseData.choices && apiResponseData.choices.length > 0) {
    const responseText = apiResponseData.choices[0].message.content;
    return new NextResponse(responseText, {
      status: 200, // Correct status code for successful responses
      headers: { 'Content-Type': 'text/plain' }
    });
  }

  // Return plain text if API response is not as expected
  return new NextResponse("Hello", {
    status: 200, // Correct status code for successful responses
    headers: { 'Content-Type': 'text/plain' }
  });
}
