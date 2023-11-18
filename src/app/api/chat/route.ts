import { notesIndex } from "@/lib/db/pinecone";
import prisma from "@/lib/db/prisma";
import openai, { getEmbedding } from "@/lib/openai";
import { auth } from "@clerk/nextjs";
import { OpenAIStream, StreamingTextResponse } from "ai";
import { ChatCompletionMessage } from "openai/resources/index.mjs";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const messages: ChatCompletionMessage[] = body.messages;

    const messagesTruncated = messages.slice(-6); // only use the last 6 messages

    const embedding = await getEmbedding(
      messagesTruncated.map((message) => message.content).join("\n"),
    );

    const { userId } = auth();

    const vectorQueryResponse = await notesIndex.query({
      vector: embedding,
      topK: 4, // num notes we send to ChatGPT; higher is more likely to find info we want, but $$$
      filter: { userId },
    });

    // pinecone returns id of notes, so we need to get the actual note via mongoDB

    const relevantNotes = await prisma.note.findMany({
      where: {
        id: {
          in: vectorQueryResponse.matches.map((match) => match.id),
        },
      },
    });

    console.log("Relevant notes found: ", relevantNotes);

    const systemMessages: ChatCompletionMessage = {
      role: "system", // Used for instructions
      content:
        "You are an intelligent note taking app. You answer the user's questions based on their existing notes." +
        "The relevant notes for this query are: \n" +
        relevantNotes
          .map((note) => `Title: ${note}\n\nContent:\n${note.content}`)
          .join("\n\n"),
    };

    // now make the request to chatGPT
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      stream: true,
      messages: [systemMessages, ...messagesTruncated],
    });

    // return to the frontend
    // vercel streaming helpers - the useChat hook knows how to read the stream
    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (error) {
    console.error(error);
    return Response.json({ error: "Internal server error" }, { status: 500 });
  }
}
