const API_BASE_URL = window.location.hostname.includes("localhost")
	? "http://127.0.0.1:8000"
	: "https://movie-recommender-backend-a9wo.onrender.com";
let currentUsername = null;

// ===== Spinner helper =====
function showSpinner(show) {
	const spinner = document.getElementById("spinner");
	if (show) {
		spinner.classList.add("show");
		spinner.style.display = "flex";
	} else {
		spinner.classList.remove("show");
		setTimeout(() => {
			spinner.style.display = "none";
		}, 300);
	}
}

// ===== Define options =====
const GENRES = [
	"Action",
	"Comedy",
	"Drama",
	"Horror",
	"Romance",
	"Sci-Fi",
	"Thriller",
];
const ACTORS = [
	"Nicolas Cage",
	"Tom Hanks",
	"Scarlett Johansson",
	"Leonardo DiCaprio",
	"Meryl Streep",
];

// ===== Initialize Page =====
window.addEventListener("DOMContentLoaded", () => {
	console.log("🖥 Page loaded");

	const genreSelect = document.getElementById("genre-options");
	GENRES.forEach((g) => {
		const opt = document.createElement("option");
		opt.value = g;
		opt.textContent = g;
		genreSelect.appendChild(opt);
	});

	const actorSelect = document.getElementById("actor-options");
	ACTORS.forEach((a) => {
		const opt = document.createElement("option");
		opt.value = a;
		opt.textContent = a;
		actorSelect.appendChild(opt);
	});

	setupMultiSelect("genre-options", "genre-select-container");
	setupMultiSelect("actor-options", "actor-select-container");

	const savedUser = localStorage.getItem("username");
	if (savedUser) {
		currentUsername = savedUser;
		document.getElementById("username").value = currentUsername;
		document.getElementById(
			"user-info"
		).textContent = `Welcome back, ${currentUsername}!`;
		document.getElementById("recommend-section").style.display = "block";
		loadHistory();
	}
});

// ===== Multi-select helper =====
function setupMultiSelect(selectId, containerId) {
	const select = document.getElementById(selectId);
	const display = document.querySelector(`#${containerId} .selected-display`);

	function updateDisplay() {
		const selected = Array.from(select.selectedOptions).map((opt) => opt.value);
		display.innerHTML =
			selected.length > 0
				? selected.map((v) => `<span>${v}</span>`).join("")
				: selectId.includes("genre")
				? "Select genres..."
				: "Select actors...";
	}

	select.addEventListener("change", updateDisplay);
	updateDisplay();

	display.addEventListener("click", () => {
		select.size = select.size === 5 ? 0 : 5;
	});
}

// ===== Login =====
document.getElementById("loginBtn").addEventListener("click", async () => {
	const usernameInput = document.getElementById("username").value.trim();
	if (!usernameInput) return alert("Enter a username!");
	currentUsername = usernameInput.toLowerCase();
	console.log(`👤 Logging in as ${currentUsername}`);

	localStorage.setItem("username", currentUsername);
	document.getElementById(
		"user-info"
	).textContent = `Welcome, ${currentUsername}!`;
	document.getElementById("recommend-section").style.display = "block";

	await loadHistory();
});

// ===== Get Recommendations =====
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUsername) return alert("Please log in first!");

	console.log("🎯 Requesting recommendations...");
	const genres = Array.from(
		document.querySelectorAll("#genre-options option:checked")
	).map((i) => i.value);
	const actors = Array.from(
		document.querySelectorAll("#actor-options option:checked")
	).map((i) => i.value);
	const type = document.getElementById("type").value;
	const recType = type === "svd" ? "collab" : type;

	const container = document.getElementById("recommendations");
	container.innerHTML = "<em>Loading recommendations...</em>";
	showSpinner(true);

	try {
		const res = await fetch(`${API_BASE_URL}/recommend/${recType}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				liked_genres: genres,
				liked_actors: actors,
				top_n: 5,
			}),
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();

		container.innerHTML = "";
		data.recommendations.forEach((m) => {
			const div = document.createElement("div");
			div.className = "movie-card";
			div.innerHTML = `<span>${m.title}</span>
        <div>
          <button onclick="sendFeedback('${m.movieId}', 'interested')">Interested</button>
          <button onclick="sendFeedback('${m.movieId}', 'watched')">Watched</button>
        </div>`;
			container.appendChild(div);
		});
	} catch (e) {
		console.error("❌ Error fetching recommendations:", e);
		container.innerHTML = "<em>Failed to load recommendations.</em>";
		alert("Failed to get recommendations. Check console for details.");
	} finally {
		showSpinner(false);
	}
});

// ===== Send Feedback =====
async function sendFeedback(movieId, type) {
	if (!currentUsername) return alert("Please log in first!");

	console.log(`✉️ Sending feedback: ${type} for movie ${movieId}`);
	showSpinner(true);
	try {
		const res = await fetch(`${API_BASE_URL}/feedback`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				movie_id: parseInt(movieId),
				interaction: type,
			}),
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		alert(`${type} saved for movie ${movieId}!`);
		await loadHistory();
	} catch (e) {
		console.error("❌ Feedback error:", e);
		alert("Failed to save feedback.");
	} finally {
		showSpinner(false);
	}
}

// ===== Load History =====
async function loadHistory() {
	if (!currentUsername) return;
	console.log("📖 Loading user history...");
	showSpinner(true);
	try {
		const res = await fetch(`${API_BASE_URL}/users/${currentUsername}/history`);
		if (!res.ok) {
			console.warn(
				"⚠️ Could not load history (likely new user). Status:",
				res.status
			);
			document.getElementById("history").innerHTML = "";
			return;
		}
		const data = await res.json();
		const div = document.getElementById("history");
		div.innerHTML = "";
		if (data.history.length === 0) {
			div.innerHTML = "<em>No history yet.</em>";
		} else {
			data.history.forEach((h) => {
				const item = document.createElement("div");
				item.className = "movie-card";
				item.innerHTML = `<span>${h.title}</span> <em>${h.interaction}</em>`;
				div.appendChild(item);
			});
		}
	} catch (e) {
		console.warn("⚠️ Failed to load history (likely new user):", e);
		document.getElementById("history").innerHTML = "<em>No history yet.</em>";
	} finally {
		showSpinner(false);
	}
}

// ===== Logout =====
document.getElementById("logoutBtn").addEventListener("click", () => {
	console.log("🚪 Logging out");
	localStorage.removeItem("username");
	currentUsername = null;
	document.getElementById("username").value = "";
	document.getElementById("recommend-section").style.display = "none";
	document.getElementById("user-info").textContent = "";
	document.getElementById("recommendations").innerHTML = "";
	document.getElementById("history").innerHTML = "";
});
