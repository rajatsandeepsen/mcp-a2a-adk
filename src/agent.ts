import { LlmAgent, MCPToolset } from "@google/adk";
import { SarvamLlm } from "../sarvam";

export const rootAgent = new LlmAgent({
	name: "an_agent",
	model: new SarvamLlm("sarvam-30b"),
	instruction:
		"You are a helpful assistant. Answer user questions using tools when needed.",
	tools: [
		new MCPToolset({
			type: "StreamableHTTPConnectionParams",
			url: "http://localhost:3000/mcp",
		}),
	],
});
