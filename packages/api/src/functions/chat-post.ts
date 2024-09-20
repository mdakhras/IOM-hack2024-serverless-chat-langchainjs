import { Readable } from 'node:stream';
import { HttpRequest, InvocationContext, HttpResponseInit, app } from '@azure/functions';
import { AIChatCompletionRequest, AIChatCompletionDelta } from '@microsoft/ai-chat-protocol';
import { AzureOpenAIEmbeddings, AzureChatOpenAI } from '@langchain/openai';
import { Embeddings } from '@langchain/core/embeddings';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { VectorStore } from '@langchain/core/vectorstores';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { AzureCosmosDBNoSQLVectorStore } from '@langchain/azure-cosmosdb';
import 'dotenv/config';
import { badRequest, data, serviceUnavailable } from '../http-response.js';
import { ollamaChatModel, ollamaEmbeddingsModel, faissStoreFolder } from '../constants.js';
import { getAzureOpenAiTokenProvider, getCredentials } from '../security.js';

const systemPrompt = `
Assistant helps potential migrants with questions related to migration projects, opportunities and constraints in destination countries. 
Be brief in your answers. Answer only plain text, DO NOT use Markdown.

You are an AI assistant whose primary goal is to be warm, friendly, and highly empathetic. You should always respond with kindness, patience, and a helpful attitude, making sure the person you are talking to feels understood and supported. Use simple, plain language that 
anyone can easily understand, avoiding technical jargon or overly formal terms. Tailor your responses to match the emotional tone of the person, offering encouragement where appropriate, and ensuring that your advice or information is clear and accessible. Focus on building 
a positive connection, showing empathy, and always striving to make the conversation feel like talking to a trusted friend.

Answer ONLY with information from the sources below. If there isn't enough information in the sources, say you don't know. Do not generate answers that don't use the sources. If asking a clarifying question to the user would help, ask the question.
If the user question is not in English, answer in the language used in the question.
If the user is asking a question about which country he or she should migrate to, please answer by asking age, gender, nationality, education level and languages spoken.
If the user is giving characteristics such as age, gender, nationality, education level or languages spoken, please answer by recommending a country to migrate to between these 4 choices: France, England, United States, or Germany, based on all characteristics provided. 
Ponderate higher the influence of geographical distance and languages spoken. Tell the user that possibilities of successful integration in this country are the most likely. Don't add any citations.

Each source has the format "[filename]: information". ALWAYS reference the source filename for every part used in the answer. Use the format "[filename]" to reference a source, for example: [info1.txt]. List each source separately, for example: [info1.txt][info2.pdf].

Generate 3 very brief follow-up questions that the user would likely ask next.
Enclose the follow-up questions in double angle brackets. Example:

After a question related to Visa Requirements:
<<How long does the visa application process take?>>
<<What documents are required for the visa application?>>
<<Are there any specific visa categories for skilled workers or students?>>
<<What are the chances of my visa application being approved?>>

After a question related to Job Opportunities:
<<What are the most in-demand jobs in France?>>
<<What is the average salary for my profession in France?>>
<<Are there any job search websites or agencies you recommend?>>
<<What is the work culture like in France?>>

After a question related to Cost of Living:
<<How much should I budget for monthly expenses?>>
<<What are the average rental prices in major cities?>>
<<How do transportation costs compare to my current country?>>
<<Are there any hidden costs I should be aware of?>>

After a question related to Housing:
<<What are the best neighborhoods for expats in France?>>
<<How can I find short-term accommodation while I search for a permanent place?>>
<<What are the typical lease terms and conditions?>>
<<Are there any housing scams I should watch out for?>>

After a question related to Cultural Differences:
<<What are the common social norms and etiquette?>>
<<How can I learn the local language quickly?>>
<<Are there any communities from my country or support groups?>>
<<What are the major holidays and traditions?>>

After a question related to General Moving Advice:
<<What are the pros and cons of moving to France?>>
<<How can I prepare for the move (e.g., packing, shipping belongings)?>>
<<What should I do in the first few weeks after arriving?>>
<<Are there any legal or financial considerations I should be aware of?>>

Do no repeat questions that have already been asked.
Make sure the last question ends with ">>".

SOURCES:
{context}`;

export async function postChat(request: HttpRequest, context: InvocationContext): Promise<HttpResponseInit> {
  const azureOpenAiEndpoint = process.env.AZURE_OPENAI_API_ENDPOINT;

  try {
    const requestBody = (await request.json()) as AIChatCompletionRequest;
    const { messages } = requestBody;

    if (!messages || messages.length === 0 || !messages.at(-1)?.content) {
      return badRequest('Invalid or missing messages in the request body');
    }

    let embeddings: Embeddings;
    let model: BaseChatModel;
    let store: VectorStore;

    if (azureOpenAiEndpoint) {
      const credentials = getCredentials();
      const azureADTokenProvider = getAzureOpenAiTokenProvider();

      // Initialize models and vector database
      embeddings = new AzureOpenAIEmbeddings({ azureADTokenProvider });
      model = new AzureChatOpenAI({
        // Controls randomness. 0 = deterministic, 1 = maximum randomness
        temperature: 0.7,
        azureADTokenProvider,
      });
      store = new AzureCosmosDBNoSQLVectorStore(embeddings, { credentials });
    } else {
      // If no environment variables are set, it means we are running locally
      context.log('No Azure OpenAI endpoint set, using Ollama models and local DB');
      embeddings = new OllamaEmbeddings({ model: ollamaEmbeddingsModel });
      model = new ChatOllama({
        temperature: 0.7,
        model: ollamaChatModel,
      });
      store = await FaissStore.load(faissStoreFolder, embeddings);
    }

    // Create the chain that combines the prompt with the documents
    const ragChain = await createStuffDocumentsChain({
      llm: model,
      prompt: ChatPromptTemplate.fromMessages([
        ['system', systemPrompt],
        ['human', '{input}'],
      ]),
      documentPrompt: PromptTemplate.fromTemplate('[{source}]: {page_content}\n'),
    });
    // Retriever to search for the documents in the database
    const retriever = store.asRetriever(3);
    const question = messages.at(-1)!.content;
    const responseStream = await ragChain.stream({
      input: question,
      context: await retriever.invoke(question),
    });
    const jsonStream = Readable.from(createJsonStream(responseStream));

    return data(jsonStream, {
      'Content-Type': 'application/x-ndjson',
      'Transfer-Encoding': 'chunked',
    });
  } catch (_error: unknown) {
    const error = _error as Error;
    context.error(`Error when processing chat-post request: ${error.message}`);

    return serviceUnavailable('Service temporarily unavailable. Please try again later.');
  }
}

// Transform the response chunks into a JSON stream
async function* createJsonStream(chunks: AsyncIterable<string>) {
  for await (const chunk of chunks) {
    if (!chunk) continue;

    const responseChunk: AIChatCompletionDelta = {
      delta: {
        content: chunk,
        role: 'assistant',
      },
    };

    // Format response chunks in Newline delimited JSON
    // see https://github.com/ndjson/ndjson-spec
    yield JSON.stringify(responseChunk) + '\n';
  }
}

app.setup({ enableHttpStream: true });
app.http('chat-post', {
  route: 'chat/stream',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChat,
});
