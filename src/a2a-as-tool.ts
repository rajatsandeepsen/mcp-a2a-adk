import { a2a } from "a2a-ai-provider";
import { generateText, tool } from "ai";
import { sarvam } from "sarvam-ai-sdk";
import z from "zod";

const weatherTool = tool({
	description: "Get the weather in a location",
	inputSchema: z.object({
		location: z.string().describe("The location to get the weather for"),
	}),
	execute: async ({ location }) => {
		const result = await generateText({
			model: a2a("http://localhost:8080/.well-known/agent-card.json"),
			prompt: `get weather at ${location}`,
		});

		return result.text;
	},
});

const { toolResults } = await generateText({
	model: sarvam("sarvam-30b"),
	tools: {
		weatherTool,
	},
	prompt: "whats the weather at kochi?",
});

console.log(toolResults);
