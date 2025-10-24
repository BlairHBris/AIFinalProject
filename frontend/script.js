// === CONFIG ===
const API_BASE_URL = window.location.hostname.includes("localhost")
	? "http://127.0.0.1:8000"
	: "https://movie-recommender-backend.onrender.com";

let currentUsername = null;

// === DOM ELEMENTS ===
const loginSection = document.getElementById("login-section");
const recommendSection = document.getElementById("recommend-section");
const usernameInput = document.getElementById("username");
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const recommendationsDiv = document.getElementById("recommendations");
const historyDiv = document.getElementById("history");

// === LOGIN ===
loginBtn.addEventListener("click", async () => {
	const username = usernameInput.value.trim();
	if (!username) return alert("Please enter a username!");

	currentUsername = username.toLowerCase();
	localStorage.setItem("username", currentUsername);

	try {
		await fetch(`${API_BASE_URL}/users/${currentUsername}/login`, {
			method: "POST",
		});

		loginSection.style.display = "none";
		recommendSection.style.display = "block";
		logoutBtn.style.display = "inline-block";

		await loadFilters();
		await loadHistory();
	} catch (err) {
		console.error("❌ Login error:", err);
		alert("Failed to log in. Please try again.");
	}
});

// === LOGOUT ===
logoutBtn.addEventListener("click", () => {
	currentUsername = null;
	localStorage.removeItem("username");
	usernameInput.value = "";
	loginSection.style.display = "block";
	recommendSection.style.display = "none";
	logoutBtn.style.display = "none";
	recommendationsDiv.innerHTML = "";
	historyDiv.innerHTML = "";
});

// === FETCH FILTER OPTIONS ===
async function loadFilters() {
	await populateSelect("genre-options", [
		"Action",
		"Adventure",
		"Animation",
		"Comedy",
		"Crime",
		"Drama",
		"Documentary",
		"Fantasy",
		"Horror",
		"Romance",
		"Sci-Fi",
		"Thriller",
		"War",
		"Western",
	]);
	await populateSelect("actor-options", [
		"Tom Hanks",
		"Leonardo DiCaprio",
		"Scarlett Johansson",
		"Meryl Streep",
		"Brad Pitt",
		"Nicolas Cage",
		"Emma Stone",
		"Johnny Depp",
	]);
	await populateSelect("movie-options", [
		"Toy Story (1995)",
		"Heat (1995)",
		"GoldenEye (1995)",
		"Casino (1995)",
		"Sense and Sensibility (1995)",
		"Four Rooms (1995)",
		"Jumanji (1995)",
	]);
}

// === POPULATE SELECT ===
function populateSelect(selectId, items) {
	const select = document.getElementById(selectId);
	select.innerHTML = "";
	items.forEach((item) => {
		const opt = document.createElement("option");
		opt.value = item;
		opt.textContent = item;
		select.appendChild(opt);
	});
}

// === FILTER SEARCH ===
function enableSelectSearch(inputId, selectId) {
	const input = document.getElementById(inputId);
	const select = document.getElementById(selectId);

	input.addEventListener("input", () => {
		const filter = input.value.toLowerCase();
		Array.from(select.options).forEach((opt) => {
			opt.style.display = opt.text.toLowerCase().includes(filter) ? "" : "none";
		});
	});
}

enableSelectSearch("genre-search", "genre-options");
enableSelectSearch("actor-search", "actor-options");
enableSelectSearch("movie-search", "movie-options");

// === CLEAR BUTTONS ===
function enableClearButton(buttonId, selectId, searchId) {
	const btn = document.getElementById(buttonId);
	const select = document.getElementById(selectId);
	const search = document.getElementById(searchId);

	btn.addEventListener("click", () => {
		Array.from(select.options).forEach((opt) => (opt.selected = false));
		search.value = "";
		Array.from(select.options).forEach((opt) => (opt.style.display = ""));
	});
}

enableClearButton("clear-genres", "genre-options", "genre-search");
enableClearButton("clear-actors", "actor-options", "actor-search");
enableClearButton("clear-movies", "movie-options", "movie-search");

// === RECOMMENDATION HANDLER ===
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUsername) return alert("Please log in first!");

	const type = document.getElementById("type").value;
	const genres = getSelectedValues("genre-options");
	const actors = getSelectedValues("actor-options");
	const movies = getSelectedValues("movie-options");

	recommendationsDiv.innerHTML = "<em>Fetching recommendations...</em>";

	try {
		const res = await fetch(`${API_BASE_URL}/recommend/${type}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				liked_genres: genres,
				liked_actors: actors,
				liked_movies: movies,
				top_n: 10,
			}),
		});

		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();

		displayRecommendations(data.recommendations || []);
		await loadHistory();
	} catch (err) {
		console.error("❌ Recommendation error:", err);
		recommendationsDiv.innerHTML = "<p>Failed to load recommendations.</p>";
	}
});

// === DISPLAY RECOMMENDATIONS ===
function displayRecommendations(recs) {
	recommendationsDiv.innerHTML = "";
	if (!recs.length) {
		recommendationsDiv.innerHTML = "<p>No recommendations found.</p>";
		return;
	}

	const table = document.createElement("table");
	table.classList.add("rec-table");

	const header = document.createElement("tr");
	["#", "Title", "Genres", "Rating"].forEach((h) => {
		const th = document.createElement("th");
		th.textContent = h;
		header.appendChild(th);
	});
	table.appendChild(header);

	recs.forEach((rec, i) => {
		const tr = document.createElement("tr");
		tr.innerHTML = `
			<td>${i + 1}</td>
			<td>${rec.title || "N/A"}</td>
			<td>${rec.genres?.join(", ") || "—"}</td>
			<td>${rec.avg_rating ? rec.avg_rating.toFixed(2) : "—"}</td>
		`;
		table.appendChild(tr);
	});

	recommendationsDiv.appendChild(table);
}

// === LOAD HISTORY ===
async function loadHistory() {
	historyDiv.innerHTML = "<em>Loading history...</em>";
	if (!currentUsername) return;

	try {
		const res = await fetch(`${API_BASE_URL}/users/${currentUsername}/history`);
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();

		if (!data.history?.length) {
			historyDiv.innerHTML = "<p>No history yet.</p>";
			return;
		}

		const list = document.createElement("ul");
		data.history.forEach((item) => {
			const li = document.createElement("li");
			li.textContent = `${item.title} — ${item.interaction}`;
			list.appendChild(li);
		});
		historyDiv.innerHTML = "";
		historyDiv.appendChild(list);
	} catch (err) {
		console.error("⚠️ Failed to load history:", err);
		historyDiv.innerHTML = "<p>Failed to load history.</p>";
	}
}

// === UTIL ===
function getSelectedValues(selectId) {
	return Array.from(document.getElementById(selectId).selectedOptions).map(
		(opt) => opt.value
	);
}

// === AUTO LOGIN ON PAGE LOAD ===
window.addEventListener("DOMContentLoaded", async () => {
	const saved = localStorage.getItem("username");
	if (saved) {
		currentUsername = saved;
		usernameInput.value = saved;
		loginSection.style.display = "none";
		recommendSection.style.display = "block";
		logoutBtn.style.display = "inline-block";

		await loadFilters();
		await loadHistory();
	}
});
