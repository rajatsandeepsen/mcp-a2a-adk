import type { LlmRequest } from "@google/adk";
import { SarvamLlm } from ".";

const llm = new SarvamLlm("sarvam-105b");

const request: LlmRequest = {
	contents: [
		{
			role: "user",
			parts: [{ text: "What is the capital of France?" }],
		},
	],
	toolsDict: {},
	liveConnectConfig: {},
};

const response = llm.generateContentAsync(request, false);

for await (const chunk of response) {
	console.log(chunk.content?.parts);
}
