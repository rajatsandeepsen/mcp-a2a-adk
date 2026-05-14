import { toA2a } from "@google/adk";
import { rootAgent } from "./agent";

const port = 8080;

const app = await toA2a(rootAgent, { port });

app.listen(port, () => {
	console.log(`Listening on port ${port}...`);
});
