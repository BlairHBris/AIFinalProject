const API_BASE_URL = "https://movie-recommender-backend.onrender.com";
let currentUser = "";

// DOM Elements
const loginSection = document.getElementById("login-section");
const recommendSection = document.getElementById("recommend-section");
const usernameInput = document.getElementById("username");
const loginButton = document.getElementById("loginButton");
const logoutButton = document.getElementById("logoutButton");
const recommendationsDiv = document.getElementById("recommendations");
const historyDiv = document.getElementById("history");
const spinner = document.getElementById("spinner");

// LOGIN
loginButton.addEventListener("click", async () => {
	const username = usernameInput.value.trim();
	if (!username) return alert("Please enter a username");
	currentUser = username;

	try {
		const res = await fetch(`${API_BASE_URL}/users/${username}/login`, {
			method: "POST",
		});
		if (!res.ok) throw new Error("Login failed");

		loginSection.style.display = "none";
		recommendSection.style.display = "block";
		logoutButton.style.display = "block";

		await loadFilters();
		await loadHistory();
	} catch (err) {
		console.error("Login error:", err);
		alert("Failed to log in.");
	}
});

// LOGOUT
logoutButton.addEventListener("click", () => {
	currentUser = "";
	loginSection.style.display = "block";
	recommendSection.style.display = "none";
	logoutButton.style.display = "none";
});

// LOAD FILTER OPTIONS
async function loadFilters() {
	await loadOptions("genre-options", "genres", "alpha");
	await loadOptions("actor-options", "actors", "alpha");
	await loadOptions("movie-options", "movies", "rating"); // top 25 movies
}

async function loadOptions(selectId, endpoint, sortType) {
	const select = document.getElementById(selectId);
	select.innerHTML = "";
	try {
		const res = await fetch(`${API_BASE_URL}/${endpoint}`);
		if (!res.ok) throw new Error(`Failed to fetch ${endpoint}`);
		let data = await res.json();

		// Sorting
		if (sortType === "alpha") data.sort((a, b) => a.localeCompare(b));
		else if (sortType === "rating") data = data.slice(0, 25); // already top 25 from backend

		data.forEach((item) => {
			const opt = document.createElement("option");
			opt.value = item;
			opt.textContent = item;
			select.appendChild(opt);
		});
	} catch (err) {
		console.error(`Failed to load ${endpoint}:`, err);
	}
}

// ENABLE DROPDOWN SEARCH
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

// CLEAR BUTTONS
function enableClearButton(buttonId, selectId, searchId) {
	const btn = document.getElementById(buttonId);
	const select = document.getElementById(selectId);
	const search = document.getElementById(searchId);

	btn.addEventListener("click", () => {
		Array.from(select.options).forEach((opt) => (opt.selected = false));
		if (search) search.value = "";
		Array.from(select.options).forEach((opt) => (opt.style.display = ""));
	});
}

enableClearButton("clear-genres", "genre-options", "genre-search");
enableClearButton("clear-actors", "actor-options", "actor-search");
enableClearButton("clear-movies", "movie-options", "movie-search");

// GET RECOMMENDATIONS
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUser) return alert("Please log in first");

	const type = document.getElementById("type").value;
	const genres = getSelectedValues("genre-options");
	const actors = getSelectedValues("actor-options");
	const movies = getSelectedValues("movie-options");

	const payload = {
		username: currentUser,
		liked_genres: genres,
		liked_actors: actors,
		liked_movies: movies,
		top_n: 10,
	};

	spinner.classList.add("show");
	recommendationsDiv.innerHTML = "";

	try {
		const res = await fetch(`${API_BASE_URL}/recommend/${type}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});

		if (!res.ok) throw new Error("Failed to fetch recommendations");
		const data = await res.json();
		displayRecommendations(data.recommendations || []);
		await loadHistory();
	} catch (err) {
		console.error("Recommendation error:", err);
		alert("Failed to fetch recommendations.");
	} finally {
		spinner.classList.remove("show");
	}
});

function displayRecommendations(recs) {
	recommendationsDiv.innerHTML = "";
	if (!recs.length) {
		recommendationsDiv.innerHTML = "<p>No recommendations found.</p>";
		return;
	}

	recs.forEach((m) => {
		const div = document.createElement("div");
		div.className = "movie-card";
		div.innerHTML = `
      <span><strong>${m.title}</strong></span>
      <em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em>
      <em>Genres: ${m.genres.join(", ")}</em>
      <em>Tags: ${m.top_tags.join(", ")}</em>
      <div class="feedback-buttons">
        <button class="feedback-btn" data-type="interested">Interested</button>
        <button class="feedback-btn" data-type="watched">Watched</button>
      </div>
    `;
		recommendationsDiv.appendChild(div);

		const [interestedBtn, watchedBtn] = div.querySelectorAll(".feedback-btn");

		async function handleFeedback(button, type) {
			const isActive = button.classList.contains("active");
			const action = isActive ? "remove" : type;
			try {
				await fetch(`${API_BASE_URL}/feedback`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						username: currentUser,
						movie_id: m.movieId,
						interaction: action,
					}),
				});
				button.classList.toggle("active");
			} catch (err) {
				console.error("Feedback error:", err);
			}
			await loadHistory();
		}

		interestedBtn.addEventListener("click", () =>
			handleFeedback(interestedBtn, "interested")
		);
		watchedBtn.addEventListener("click", () =>
			handleFeedback(watchedBtn, "watched")
		);
	});
}

// LOAD HISTORY
async function loadHistory() {
	historyDiv.innerHTML = "";
	try {
		const res = await fetch(`${API_BASE_URL}/users/${currentUser}/history`);
		if (!res.ok) throw new Error("Failed to load history");
		const data = await res.json();
		if (!data.history?.length) {
			historyDiv.innerHTML = "<p>No history yet.</p>";
			return;
		}

		data.history.forEach((h) => {
			const div = document.createElement("div");
			div.className = "movie-card";
			div.innerHTML = `<span>${h.title}</span><em>${
				h.interaction
			}</em><em>Genres: ${h.genres.join(", ")}</em>`;
			historyDiv.appendChild(div);
		});
	} catch (err) {
		console.error("Failed to load history:", err);
	}
}

// UTIL
function getSelectedValues(selectId) {
	return Array.from(document.getElementById(selectId).selectedOptions).map(
		(opt) => opt.value
	);
}
