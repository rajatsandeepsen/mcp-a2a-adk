import { LlmAgent, MCPToolset } from "@google/adk";
import { SarvamLlm } from "../sarvam";
import { getWeatherTool } from "./tools";

export const rootAgent = new LlmAgent({
	name: "an_agent",
	model: new SarvamLlm("sarvam-30b", {
		// log: true,
	}),
	instruction:
		"You are a helpful assistant. Answer user questions using tools when needed.",
	tools: [
		// getWeatherTool,
		new MCPToolset({
			type: "StreamableHTTPConnectionParams",
			url: "http://localhost:3000/mcp",
		}),
	],
});
