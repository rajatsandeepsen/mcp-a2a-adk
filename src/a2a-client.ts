import { a2a } from "a2a-ai-provider";
import { generateText } from "ai";

const result = await generateText({
	model: a2a("http://localhost:8080/.well-known/agent-card.json"),
	prompt: "What weather at kottayam",
});

console.log(JSON.stringify(result, null, 2));
