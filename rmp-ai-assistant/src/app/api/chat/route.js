import { NextResponse } from 'next/server';

const sysPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`;

const LLAMA_API_KEY = process.env.LLAMA_API_KEY;

export async function POST(req) {
  const data = await req.json();

  try {
    const completion = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${LLAMA_API_KEY}`, 
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

    // Check if the API response has choices and return the first choice's message content
    if (apiResponseData.choices && apiResponseData.choices.length > 0) {
      const responseText = apiResponseData.choices[0].message.content;
      return new NextResponse(responseText, {
        status: 200,
        headers: { 'Content-Type': 'text/plain' }
      });
    }

    // Return default message if API response is not as expected
    return new NextResponse("Hello", {
      status: 200,
      headers: { 'Content-Type': 'text/plain' }
    });
  } catch (error) {
    // Handle errors and return an error message
    return new NextResponse("An error occurred", {
      status: 500,
      headers: { 'Content-Type': 'text/plain' }
    });
  }
}
