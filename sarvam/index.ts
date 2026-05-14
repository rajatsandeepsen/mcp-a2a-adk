import {
	BaseLlm,
	type BaseLlmConnection,
	type LlmRequest,
	type LlmResponse,
} from "@google/adk";
import type { Content } from "@google/genai";

/**
 * Configuration for Sarvam LLM
 */
interface SarvamConfig {
	apiKey?: string;
	apiUrl?: string;
	log?: boolean;
}

/**
 * Sarvam API request structure
 */
interface SarvamChatCompletionRequest {
	messages: Array<{
		role: "user" | "assistant" | "system" | "tool";
		content: string;
		tool_call_id?: string;
	}>;
	model: string;
	temperature?: number;
	top_p?: number;
	max_tokens?: number;
	stream?: boolean;
	stop?: string | string[];
	reasoning_effort?: "low" | "medium" | "high";
	presence_penalty?: number;
	frequency_penalty?: number;
	seed?: number;
	wiki_grounding?: boolean;
	tools?: Array<{
		type: "function";
		function: {
			name: string;
			description?: string;
			parameters?: object;
		};
	}>;
	tool_choice?: "none" | "auto" | "required" | object;
}

/**
 * Sarvam API response structure
 */
interface SarvamChatCompletionResponse {
	id: string;
	object: "chat.completion" | "chat.completion.chunk";
	created: number;
	model: string;
	choices: Array<{
		index: number;
		message: {
			role: "assistant";
			content: string | null;
			tool_calls?: Array<{
				id: string;
				type: "function";
				function: {
					name: string;
					arguments: string;
				};
			}>;
		};
		finish_reason: "stop" | "length" | "tool_calls" | "content_filter";
	}>;
	usage?: {
		prompt_tokens: number;
		completion_tokens: number;
		total_tokens: number;
	};
}

interface SarvamStreamChunk {
	choices: Array<{
		delta?: {
			content?: string;
			tool_calls?: Array<{
				index: number;
				id: string;
				type: "function";
				function: {
					name: string;
					arguments: string;
				};
			}>;
		};
		finish_reason?:
			| "stop"
			| "length"
			| "tool_calls"
			| "content_filter"
			| "function_call"
			| null;
	}>;
}

type SarvamModels = "sarvam-105b" | "sarvam-30b" | "sarvam-m";

/**
 * SarvamLlm - Implementation of BaseLlm for Sarvam AI models
 *
 * Supports:
 * - sarvam-105b (128K context)
 * - sarvam-30b (64K context)
 * - sarvam-m (24B, legacy)
 */
export class SarvamLlm extends BaseLlm {
	private apiKey: string;
	private apiUrl: string;
	private log: boolean;

	/**
	 * List of supported Sarvam models
	 */
	static override readonly supportedModels = [
		"sarvam-105b",
		"sarvam-30b",
		"sarvam-m",
		/^sarvam-.*/,
	];

	/**
	 * Constructor for SarvamLlm
	 * @param model The name of the Sarvam model to use
	 * @param config Configuration including API key
	 */
	constructor(model: SarvamModels, config?: SarvamConfig) {
		super({ model });
		this.apiKey = config?.apiKey || (process.env.SARVAM_API_KEY as string);
		this.apiUrl = config?.apiUrl || "https://api.sarvam.ai/v1";
		this.log = config?.log || false;
	}

	/**
	 * Generates content asynchronously from the Sarvam API
	 * @param llmRequest The LLM request containing contents, tools, and config
	 * @param stream Whether to stream the response (optional, default: false)
	 * @param abortSignal Signal to abort the request (optional)
	 * @returns An async generator yielding LlmResponse objects
	 */
	async *generateContentAsync(
		llmRequest: LlmRequest,
		stream: boolean = false,
		abortSignal?: AbortSignal,
	): AsyncGenerator<LlmResponse, void> {
		try {
			// Convert Google GenAI Content format to Sarvam chat format
			const messages = this.convertContentsToSarvamMessages(
				llmRequest.contents,
			);

			// Build Sarvam request
			const sarvamRequest: SarvamChatCompletionRequest = {
				messages,
				model: this.model,
				stream: stream,
				temperature: llmRequest.config?.temperature as number | undefined,
				top_p: llmRequest.config?.topP as number | undefined,
				max_tokens: llmRequest.config?.maxOutputTokens as number | undefined,
			};

			// Add tools if present
			if (Object.keys(llmRequest.toolsDict).length > 0) {
				sarvamRequest.tools = this.convertToolsToSarvamFormat(
					llmRequest.toolsDict,
				);

				if (llmRequest.config?.toolConfig) {
					const toolConfig = llmRequest.config.toolConfig as any;
					if (toolConfig.functionCallingConfig?.mode) {
						sarvamRequest.tool_choice = this.mapToolChoiceMode(
							toolConfig.functionCallingConfig.mode,
						);
					}
				}
			}

			if (this.log) {
				console.log(
					"[SarvamLlm] Request:",
					JSON.stringify(sarvamRequest, null, 2),
				);
			}
			const response = await fetch(`${this.apiUrl}/chat/completions`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${this.apiKey}`,
					"api-subscription-key": this.apiKey,
					...this.trackingHeaders,
				},
				body: JSON.stringify(sarvamRequest),
				signal: abortSignal,
			});

			if (this.log) {
				console.log("[SarvamLlm] API:", response.status, response.statusText);
			}

			if (!response.ok) {
				const errorData = (await response.json()) as any;
				yield {
					errorCode: errorData.error?.code || response.status.toString(),
					errorMessage:
						errorData.error?.message || `API Error: ${response.statusText}`,
				};
				return;
			}

			if (stream && response.body) {
				if (this.log) {
					console.log("[SarvamLlm] streaming: started");
				}

				// Handle streaming response
				yield* this.handleStreamingResponse(response.body);
			} else {
				// Handle non-streaming response
				const data = (await response.json()) as SarvamChatCompletionResponse;

				if (this.log) {
					console.log("[SarvamLlm] non streaming", data);
				}

				yield this.convertSarvamResponseToLlmResponse(data);
			}
		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				yield { interrupted: true };
				return;
			}

			yield {
				errorCode: "REQUEST_ERROR",
				errorMessage: error instanceof Error ? error.message : String(error),
				interrupted: abortSignal?.aborted,
			};
		}
	}

	/**
	 * Creates a live connection to the Sarvam API
	 * Note: Sarvam API currently doesn't support live connections through this SDK
	 * This is a placeholder for future implementations
	 */
	async connect(llmRequest: LlmRequest): Promise<BaseLlmConnection> {
		throw new Error(
			"Live connections are not yet supported for Sarvam LLM. Please use generateContentAsync for streaming.",
		);
	}

	/**
	 * Converts Google GenAI Content format to Sarvam chat message format
	 */
	private convertContentsToSarvamMessages(
		contents: Content[],
	): SarvamChatCompletionRequest["messages"] {
		return contents.map((content) => {
			// Check if any part is a functionResponse
			const functionResponsePart = content.parts?.find(
				(part: any) => part.functionResponse,
			);

			if (functionResponsePart) {
				// Create a tool message for function responses
				const message: SarvamChatCompletionRequest["messages"][0] = {
					role: "tool",
					content: JSON.stringify(
						functionResponsePart.functionResponse?.response,
					),
					tool_call_id: functionResponsePart.functionResponse?.name,
				};
				return message;
			}

			// Process regular text messages
			let role = content.role || "user";
			// Map 'model' role from Google GenAI to 'assistant' for Sarvam API
			if (role === "model") {
				role = "assistant";
			}
			let text = "";

			// Extract text from parts
			if (content.parts && Array.isArray(content.parts)) {
				text = content.parts
					.map((part: any) => {
						if (typeof part.text === "string") {
							return part.text;
						}
						return "";
					})
					.join("");
			}

			const message: SarvamChatCompletionRequest["messages"][0] = {
				role: role as "user" | "assistant" | "system" | "tool",
				content: text,
			};

			return message;
		});
	}

	/**
	 * Converts BaseTool format to Sarvam tools format
	 */
	private convertToolsToSarvamFormat(toolsDict: {
		[key: string]: any;
	}): SarvamChatCompletionRequest["tools"] {
		return Object.values(toolsDict).map((tool) => ({
			type: "function" as const,
			function: {
				name: tool.name || "",
				description: tool.description,
				parameters: tool.inputSchema,
			},
		}));
	}

	/**
	 * Maps Google tool choice modes to Sarvam tool_choice values
	 */
	private mapToolChoiceMode(mode: string) {
		switch (mode) {
			case "AUTO":
				return "auto";
			case "ANY":
				return "required";
			case "NONE":
				return "none";
			default:
				return "auto";
		}
	}

	/**
	 * Handles streaming response from Sarvam API
	 */
	private async *handleStreamingResponse(
		body: ReadableStream<Uint8Array>,
	): AsyncGenerator<LlmResponse, void> {
		const reader = body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });

				// Process complete SSE messages
				const lines = buffer.split("\n");
				buffer = lines.pop() || ""; // Keep incomplete line in buffer

				for (const line of lines) {
					if (line.startsWith("data: ")) {
						const data = line.slice(6).trim();

						if (data === "[DONE]") {
							yield { turnComplete: true };
							continue;
						}

						if (data) {
							try {
								const chunk = JSON.parse(data) as SarvamStreamChunk;
								if (this.log) {
									console.log(
										"[SarvamLlm] stream chunk:",
										JSON.stringify(chunk.choices[0]?.delta, null, 2),
									);
								}
								yield this.convertStreamChunkToLlmResponse(chunk);
							} catch (e) {
								// Skip malformed JSON
							}
						}
					}
				}
			}

			// Process any remaining buffer
			if (buffer.startsWith("data: ")) {
				const data = buffer.slice(6).trim();
				if (data && data !== "[DONE]") {
					try {
						const chunk = JSON.parse(data) as SarvamStreamChunk;
						if (this.log) {
							console.log(
								"[SarvamLlm] stream chunk (final):",
								JSON.stringify(chunk, null, 2),
							);
						}
						yield this.convertStreamChunkToLlmResponse(chunk);
					} catch (e) {
						// Skip malformed JSON
					}
				}
			}
		} finally {
			reader.releaseLock();
		}
	}

	/**
	 * Converts Sarvam streaming chunk to LlmResponse
	 */
	private convertStreamChunkToLlmResponse(
		chunk: SarvamStreamChunk,
	): LlmResponse {
		const choice = chunk.choices?.[0];

		if (!choice) {
			return {};
		}

		const delta = choice.delta;
		const response: LlmResponse = {
			partial: !choice.finish_reason,
			turnComplete: choice.finish_reason ? true : false,
			finishReason: choice.finish_reason
				? (choice.finish_reason.toUpperCase() as LlmResponse["finishReason"])
				: undefined,
		};

		// Handle text content
		if (delta?.content) {
			response.content = {
				role: "model",
				parts: [{ text: delta.content }],
			};
		}

		// Handle tool calls
		if (delta?.tool_calls && delta.tool_calls.length > 0) {
			response.content = {
				role: "model",
				parts: delta.tool_calls.map((toolCall) => ({
					functionCall: {
						name: toolCall.function.name,
						args: JSON.parse(toolCall.function.arguments),
					},
				})),
			};
		}

		return response;
	}

	/**
	 * Converts Sarvam response to LlmResponse
	 */
	private convertSarvamResponseToLlmResponse(
		sarvamResponse: SarvamChatCompletionResponse,
	): LlmResponse {
		const choice = sarvamResponse.choices?.[0];

		if (!choice) {
			return {
				errorCode: "NO_CHOICES",
				errorMessage: "No choices in response",
			};
		}

		const response: LlmResponse = {
			finishReason: choice.finish_reason
				? (choice.finish_reason.toUpperCase() as LlmResponse["finishReason"])
				: undefined,
		};

		// Handle text content
		if (choice.message?.content) {
			response.content = {
				role: "model",
				parts: [{ text: choice.message.content }],
			};
		}

		// Handle tool calls
		if (choice.message?.tool_calls && choice.message.tool_calls.length > 0) {
			response.content = {
				role: "model",
				parts: choice.message.tool_calls.map((toolCall) => ({
					functionCall: {
						name: toolCall.function.name,
						args: JSON.parse(toolCall.function.arguments),
					},
				})),
			};
		}

		// Add usage metadata
		if (sarvamResponse.usage) {
			response.usageMetadata = {
				promptTokenCount: sarvamResponse.usage.prompt_tokens,
				candidatesTokenCount: sarvamResponse.usage.completion_tokens,
				totalTokenCount: sarvamResponse.usage.total_tokens,
			};
		}

		return response;
	}
}
