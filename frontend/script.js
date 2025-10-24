const API_BASE_URL = "https://movie-recommender-backend-a9wo.onrender.com";
let currentUser = "";

// ------------------- DOM ELEMENTS -------------------
const loginSection = document.getElementById("login-section");
const recommendSection = document.getElementById("recommend-section");
const usernameInput = document.getElementById("username");
const loginButton = document.getElementById("loginButton");
const logoutButton = document.getElementById("logoutButton");
const recommendationsDiv = document.getElementById("recommendations");
const historyDiv = document.getElementById("history");
const spinner = document.getElementById("spinner");

// ------------------- INITIALIZE FILTERS -------------------
window.addEventListener("DOMContentLoaded", async () => {
	await loadFilters();
});

// ------------------- LOGIN -------------------
loginButton.addEventListener("click", async () => {
	const username = usernameInput.value.trim();
	if (!username) return alert("Please enter a username");
	currentUser = username;

	const userType =
		document.querySelector('input[name="userType"]:checked')?.value ||
		"existing";

	try {
		// Update UI visibility
		loginSection.style.display = "none";
		recommendSection.style.display = "block";
		logoutButton.style.display = "block";

		// Load history/greeting
		if (userType === "new") {
			historyDiv.innerHTML = "<p>Welcome, new user!</p>";
		} else {
			await loadHistory();
		}
	} catch (err) {
		console.error("Login error:", err);
		alert("Failed to log in.");
	}
});

// ------------------- LOGOUT -------------------
logoutButton.addEventListener("click", () => {
	currentUser = "";
	loginSection.style.display = "block";
	recommendSection.style.display = "none";
	logoutButton.style.display = "none";
	historyDiv.innerHTML = "";
	recommendationsDiv.innerHTML = "";
});

// ------------------- FILTER OPTIONS -------------------
async function loadFilters() {
	await loadOptions("genre-options", "genres", "alpha");
	await loadOptions("movie-options", "movies", "rating");
}

async function loadOptions(selectId, endpoint, sortType) {
	const select = document.getElementById(selectId);
	select.innerHTML = "";
	try {
		const res = await fetch(`${API_BASE_URL}/${endpoint}`);

		if (!res.ok) {
			const errorDetail = await res
				.json()
				.catch(() => ({ detail: "Unknown error" }));
			throw new Error(
				`Failed to fetch ${endpoint} (${res.status}): ${errorDetail.detail}`
			);
		}

		let data = await res.json();

		if (sortType === "alpha") data.sort((a, b) => a.localeCompare(b));
		else if (sortType === "rating") data = data.slice(0, 25);

		data.forEach((item) => {
			const opt = document.createElement("option");
			opt.value = item;
			opt.textContent = item;
			select.appendChild(opt);
		});
		console.log(`✅ Loaded ${selectId.split("-")[0]} options.`);
	} catch (err) {
		console.error(`❌ Failed to load ${endpoint}:`, err);
		select.innerHTML = `<option disabled>Error loading data</option>`;
	}
}

// ------------------- GET RECOMMENDATIONS -------------------
document.getElementById("getRecsBtn").addEventListener("click", async () => {
	if (!currentUser) return alert("Please log in first");

	const type = document.getElementById("type").value;
	const genres = getSelectedValues("genre-options");
	const movies = getSelectedValues("movie-options");

	const payload = {
		username: currentUser,
		liked_genres: genres,
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
	} catch (err) {
		console.error("Recommendation error:", err);
		alert("Failed to fetch recommendations.");
	} finally {
		spinner.classList.remove("show");
	}
});

// ------------------- DISPLAY RECOMMENDATIONS -------------------
function displayRecommendations(recs) {
	recommendationsDiv.innerHTML = "";
	if (!recs.length) {
		recommendationsDiv.innerHTML = "<p>No recommendations found.</p>";
		return;
	}

	recs.forEach((m) => {
		const div = document.createElement("div");
		div.className = "movie-card";
		div.setAttribute("data-movie-id", m.movieId); // Added ID for easier hiding
		div.innerHTML = `
            <span><strong>${m.title}</strong></span>
            <em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em>
            <em>Genres: ${m.genres.join(", ")}</em>
            <em>Tags: ${m.top_tags.join(", ")}</em>
            <div class="feedback-buttons">
                <button class="feedback-btn" data-type="interested">Interested</button>
                <button class="feedback-btn" data-type="watched">Watched and Liked</button>
                <button class="feedback-btn not-interested-btn" data-type="not_interested">Not Interested</button>
            </div>
        `;
		recommendationsDiv.appendChild(div);

		const [interestedBtn, watchedBtn, notInterestedBtn] =
			div.querySelectorAll(".feedback-btn");

		const setupFeedback = (button, type) => {
			button.addEventListener("click", async () => {
				const isActive = button.classList.contains("active");

				if (type === "not_interested") {
					// Not Interested: Send permanent flag and hide card
					await sendFeedback(m.movieId, "not_interested");
					div.style.display = "none";
					await loadHistory();
					return;
				}

				// Interested/Watched Logic (Toggle)
				const action = isActive ? "remove" : type;

				try {
					await sendFeedback(m.movieId, action);

					// Update buttons
					button.classList.toggle("active", !isActive);

					// Optional: Deactivate other button if necessary (e.g., watching implies interested, so we might only show one active)
					// (For simplicity, we let the user select both, but the history will show the last action)

					updateHistoryLocal(m, action);
				} catch (err) {
					console.error("Feedback error:", err);
				}
			});
		};

		setupFeedback(interestedBtn, "interested");
		setupFeedback(watchedBtn, "watched");
		setupFeedback(notInterestedBtn, "not_interested");
	});
}

// Helper function to centralize feedback API call
async function sendFeedback(movieId, interaction) {
	if (!currentUser) return;
	try {
		const res = await fetch(`${API_BASE_URL}/feedback`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUser,
				movie_id: movieId,
				interaction: interaction,
			}),
		});
		if (!res.ok) throw new Error("Failed to send feedback");
	} catch (err) {
		console.error("Feedback API error:", err);
		alert(`Failed to record interaction: ${interaction}.`);
	}
}

// ------------------- LOAD HISTORY -------------------
async function loadHistory() {
	historyDiv.innerHTML = "";
	if (!currentUser) return;

	try {
		const res = await fetch(`${API_BASE_URL}/users/${currentUser}/history`);
		if (!res.ok) throw new Error("Failed to load history");
		const data = await res.json();

		if (data.is_new_user) {
			historyDiv.innerHTML = "<p>Welcome, new user!</p>";
			return;
		}

		displayHistory(data.history);
	} catch (err) {
		console.error("Failed to load history:", err);
	}
}

// ------------------- DISPLAY HISTORY -------------------
function displayHistory(history) {
	historyDiv.innerHTML = "";
	if (!history || history.length === 0) {
		historyDiv.innerHTML = "<p>No history yet.</p>";
		return;
	}

	history.forEach((h) => {
		const div = document.createElement("div");
		div.className = "movie-card";
		div.innerHTML = `
            <span><strong>${h.title}</strong></span>
            <em>Interaction: ${h.interaction}</em>
            <em>Genres: ${h.genres.join(", ")}</em>
        `;
		historyDiv.appendChild(div);
	});
}

// ------------------- LOCAL HISTORY UPDATE -------------------
function updateHistoryLocal(movie, interaction) {
	const existing = Array.from(historyDiv.querySelectorAll(".movie-card")).find(
		(div) => div.querySelector("span").textContent.includes(movie.title)
	);

	if (interaction === "remove") {
		if (existing) existing.remove();
		return;
	}

	if (existing) {
		const ems = existing.querySelectorAll("em");
		if (ems.length > 0) ems[0].textContent = `Interaction: ${interaction}`;
	} else {
		const div = document.createElement("div");
		div.className = "movie-card";
		div.innerHTML = `
            <span><strong>${movie.title}</strong></span>
            <em>Interaction: ${interaction}</em>
            <em>Genres: ${movie.genres.join(", ")}</em>
        `;
		historyDiv.appendChild(div);
	}
}

// ------------------- UTIL -------------------
function getSelectedValues(selectId) {
	return Array.from(document.getElementById(selectId).selectedOptions).map(
		(opt) => opt.value
	);
}
