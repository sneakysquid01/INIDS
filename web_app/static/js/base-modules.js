import { GlobalState } from "/static/js/core/global-state.js";
import { Socket } from "/static/js/core/socket-manager.js";
import { HttpClient_Instance } from "/static/js/core/http-client.js";

window.GlobalState = GlobalState;
window.Socket = Socket;
window.HttpClient = HttpClient_Instance;

console.log("[INIDS] Core modules loaded");
