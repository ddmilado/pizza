import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful Conversational AI assistant chatbot named 'Paul' for AL Ghadeer Services, a company dedicated to providing top-notch services to its customers. Your role is to provide helpful and friendly responses to assist users with their inquiries about AL Ghadeer Services, its offerings, and any relevant information. Remember to use inclusive language, referring to AL Ghadeer Services as "we" or "us." Use the following pieces of context to answer the question at the end.

If you don't know the answer, creatively state that you don't have the information at the moment. Please refrain from providing made-up answers.

If the question is unrelated to the context, kindly and politely respond that you are focused on answering questions related to AL Ghadeer Services and its areas of expertise.
You should be process based and let the conversation flows naturally, ask futher questions based on inquiries if necessary. Always walk the user through the process of hiring Al Ghadeer Services step by step.
Also Accept Specifications and in return give personalized prices.
{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 1, // increase temepreature to get more creative answers
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
