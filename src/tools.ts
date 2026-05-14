import { FunctionTool } from "@google/adk";
import z from "zod";

export const getWeatherTool = new FunctionTool({
	name: "get_weather",
	description: "Retrieves the current weather report for a specified location.",
	parameters: z.object({
		location: z.string().describe("The name of the location."),
	}),
	execute: async ({ location }) => {
		const temperature = 72 + Math.floor(Math.random() * 21) - 10;
		console.log("Called", temperature);
		return {
			location,
			temperature,
		};
	},
});
