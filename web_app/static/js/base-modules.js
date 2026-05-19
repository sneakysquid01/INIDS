// Enable verbose logging by setting `window.__INIDS_DEBUG__ = true` before this script runs.
import { GlobalState } from "/static/js/core/global-state.js";
import { Socket } from "/static/js/core/socket-manager.js";
import { HttpClient_Instance } from "/static/js/core/http-client.js";

window.GlobalState = GlobalState;
window.Socket = Socket;
window.HttpClient = HttpClient_Instance;

// Bootstrap JWT from session cookie set by /login
const _jwtCookie = document.cookie.split(";").map(c => c.trim()).find(c => c.startsWith("inids_jwt="));
if (_jwtCookie) {
    HttpClient_Instance.setToken(_jwtCookie.slice("inids_jwt=".length), 3600);
}
