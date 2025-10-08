const API_BASE = "http://127.0.0.1:8000";
let currentUsername = null;

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
	// Populate genre checkboxes
	const genreContainer = document.getElementById("genre-options");
	GENRES.forEach((g) => {
		const label = document.createElement("label");
		label.innerHTML = `<input type="checkbox" value="${g}" /> ${g}`;
		genreContainer.appendChild(label);
	});

	// Populate actor checkboxes
	const actorContainer = document.getElementById("actor-options");
	ACTORS.forEach((a) => {
		const label = document.createElement("label");
		label.innerHTML = `<input type="checkbox" value="${a}" /> ${a}`;
		actorContainer.appendChild(label);
	});

	// Auto-login if username saved
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

// ===== Login / Create User =====
document.getElementById("loginBtn").addEventListener("click", async () => {
	const usernameInput = document.getElementById("username").value.trim();
	if (!usernameInput) return alert("Enter a username!");
	currentUsername = usernameInput.toLowerCase();

	const mode = document.querySelector('input[name="userMode"]:checked').value;

	if (mode === "new") {
		await fetch(`${API_BASE}/users/new`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ name: currentUsername }),
		});
	} else {
		const res = await fetch(`${API_BASE}/users/new`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ name: currentUsername }),
		});
		const data = await res.json();
		if (data.message === "New user created") {
			alert("Username not found. Created new user automatically.");
		}
	}

	localStorage.setItem("username", currentUsername);
	document.getElementById(
		"user-info"
	).textContent = `Welcome, ${currentUsername}!`;
	document.getElementById("recommend-section").style.display = "block";

	loadHistory();
});

// ===== Get Recommendations =====
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUsername) return alert("Please log in first!");

	const genres = Array.from(
		document.querySelectorAll("#genre-options input:checked")
	).map((i) => i.value);
	const actors = Array.from(
		document.querySelectorAll("#actor-options input:checked")
	).map((i) => i.value);
	const type = document.getElementById("type").value;

	const recType = type === "svd" ? "collab" : type; // Switch SVD -> collab

	const res = await fetch(`${API_BASE}/recommend/${recType}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			username: currentUsername,
			liked_genres: genres,
			liked_actors: actors,
			top_n: 5,
		}),
	});

	const data = await res.json();
	const container = document.getElementById("recommendations");
	container.innerHTML = "";

	data.recommendations.forEach((m) => {
		const div = document.createElement("div");
		div.className = "movie-card";
		div.innerHTML = `
      <span>${m.title}</span>
      <div>
        <button onclick="sendFeedback('${m.movieId}', 'interested')">Interested</button>
        <button onclick="sendFeedback('${m.movieId}', 'watched')">Watched</button>
      </div>
    `;
		container.appendChild(div);
	});
});

// ===== Send Feedback =====
async function sendFeedback(movieId, type) {
	if (!currentUsername) return alert("Please log in first!");

	await fetch(`${API_BASE}/feedback`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			username: currentUsername,
			movie_id: parseInt(movieId),
			interaction: type,
		}),
	});

	alert(`${type} saved for movie ${movieId}!`);
	loadHistory();
}

// ===== Load History =====
async function loadHistory() {
	if (!currentUsername) return;

	const res = await fetch(`${API_BASE}/users/${currentUsername}/history`);
	const data = await res.json();
	const div = document.getElementById("history");
	div.innerHTML = "";

	data.history.forEach((h) => {
		const item = document.createElement("div");
		item.className = "movie-card";
		item.innerHTML = `<span>${h.title}</span> <em>${h.interaction}</em>`;
		div.appendChild(item);
	});
}

// ===== Logout =====
document.getElementById("logoutBtn").addEventListener("click", () => {
	localStorage.removeItem("username");
	currentUsername = null;
	document.getElementById("username").value = "";
	document.getElementById("recommend-section").style.display = "none";
	document.getElementById("user-info").textContent = "";
	document.getElementById("recommendations").innerHTML = "";
	document.getElementById("history").innerHTML = "";
});
