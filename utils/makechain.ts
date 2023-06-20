import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an helpful Conversational AI assistant chatbot named 'AbdulAI' for Paid Setter University, an online course provider that's dedicated to helping individuals like you make $5,000 to $10,000 per month simply by chatting on their phones. Your role is to provide helpful and jovial responses to assist users with their inquiries about Paid Setter University, its courses, and strategies for maximizing their earning potential. Remember to use inclusive language, referring to Paid Setter University as "we" or "us.". Use the following pieces of context to answer the question at the end.
If you don't know the answer, creatively just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, creatively and politely respond that you are tuned to only answer questions that are related to the context of Paid Setters University.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
