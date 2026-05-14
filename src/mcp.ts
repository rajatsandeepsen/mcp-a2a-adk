import {
	McpServer,
	WebStandardStreamableHTTPServerTransport,
} from "@modelcontextprotocol/server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import z from "zod";

// Create the MCP server
const server = new McpServer({
	name: "hono-webstandard-mcp-server",
	version: "1.0.0",
});

// Register a weather tool
server.registerTool(
	"get_weather",
	{
		title: "Weather Tool",
		description: "Get current weather for a location",
		inputSchema: z.object({
			location: z.string().describe("City name or location"),
		}),
	},
	async ({ location }) => {
		const data = {
			location,
			temperature: Math.floor(Math.random() * 30) + 5,
		};

		console.log(data);

		return {
			structuredContent: data,
			content: [
				{
					type: "text",
					text: `Weather in ${data.location}: ${data.temperature}°C,`,
				},
			],
		};
	},
);

// Create a stateless transport (no options = no session management)
const transport = new WebStandardStreamableHTTPServerTransport();

const app = new Hono();

// Enable CORS for all origins
app.use(
	"*",
	cors({
		origin: "*",
		allowMethods: ["GET", "POST", "DELETE", "OPTIONS"],
		allowHeaders: [
			"Content-Type",
			"mcp-session-id",
			"Last-Event-ID",
			"mcp-protocol-version",
		],
		exposeHeaders: ["mcp-session-id", "mcp-protocol-version"],
	}),
);

// Health check endpoint
app.get("/health", (c) => c.json({ status: "ok" }));

// MCP endpoint
app.all("/mcp", (c) => transport.handleRequest(c.req.raw));

await server.connect(transport);

export default app;
