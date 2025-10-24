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
		setTimeout(() => (spinner.style.display = "none"), 300);
	}
}

// ===== Initialize Page =====
window.addEventListener("DOMContentLoaded", async () => {
	console.log("🖥 Page loaded");

	const savedUser = localStorage.getItem("username");
	if (savedUser) {
		currentUsername = savedUser;
		document.getElementById("username").value = currentUsername;
		document.getElementById(
			"user-info"
		).textContent = `Welcome back, ${currentUsername}!`;
		document.getElementById("recommend-section").style.display = "block";
		await loadHistory();
	}

	await populateOptions();
	setupMultiSelect("genre-options", "genre-select-container");
	setupMultiSelect("actor-options", "actor-select-container");
	setupMultiSelect("movie-options", "movie-select-container");
});

// ===== Populate dropdowns =====
async function populateOptions() {
	const GENRES = [
		"Action",
		"Adventure",
		"Animation",
		"Children",
		"Comedy",
		"Crime",
		"Documentary",
		"Drama",
		"Fantasy",
		"Film-Noir",
		"Horror",
		"Musical",
		"Mystery",
		"Romance",
		"Sci-Fi",
		"Thriller",
		"War",
		"Western",
	];
	const ACTORS = [
		"Nicolas Cage",
		"Tom Hanks",
		"Scarlett Johansson",
		"Leonardo DiCaprio",
		"Meryl Streep",
		"Brad Pitt",
		"Emma Watson",
		"Johnny Depp",
	];
	const MOVIES = [
		"Toy Story (1995)",
		"Jumanji (1995)",
		"Grumpier Old Men (1995)",
		"Waiting to Exhale (1995)",
		"Father of the Bride Part II (1995)",
		"Heat (1995)",
		"Sabrina (1995)",
		"Tom and Huck (1995)",
		"Sudden Death (1995)",
		"GoldenEye (1995)",
		"American President, The (1995)",
		"Dracula: Dead and Loving It (1995)",
		"Balto (1995)",
		"Nixon (1995)",
		"Cutthroat Island (1995)",
		"Casino (1995)",
		"Sense and Sensibility (1995)",
		"Four Rooms (1995)",
		"Ace Ventura: When Nature Calls (1995)",
		"Money Train (1995)",
	];

	const addOptions = (id, arr) => {
		const sel = document.getElementById(id);
		arr.forEach((v) => {
			const opt = document.createElement("option");
			opt.value = v;
			opt.textContent = v;
			sel.appendChild(opt);
		});
	};
	addOptions("genre-options", GENRES);
	addOptions("actor-options", ACTORS);
	addOptions("movie-options", MOVIES);
}

// ===== Multi-select helper =====
function setupMultiSelect(selectId, containerId) {
	const select = document.getElementById(selectId);
	const display = document.querySelector(`#${containerId} .selected-display`);

	const updateDisplay = () => {
		const selected = Array.from(select.selectedOptions).map((o) => o.value);
		display.innerHTML = selected.length
			? selected.map((v) => `<span>${v}</span>`).join("")
			: selectId.includes("genre")
			? "Select genres..."
			: selectId.includes("actor")
			? "Select actors..."
			: "Select movies...";
	};

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
	const movies = Array.from(
		document.querySelectorAll("#movie-options option:checked")
	).map((i) => i.value);
	const type = document.getElementById("type").value;

	const container = document.getElementById("recommendations");
	container.innerHTML = "<em>Loading recommendations...</em>";
	showSpinner(true);

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
		container.innerHTML = "";

		data.recommendations.forEach((m) => {
			const div = document.createElement("div");
			div.className = "movie-card";
			div.innerHTML = `
        <span>${m.title}</span>
        <div>
          <em>Rating: ${m.avg_rating?.toFixed(2) ?? "N/A"}</em><br>
          <em>Genres: ${m.genres.join(", ")}</em><br>
          <em>Tags: ${m.top_tags.join(", ")}</em><br>
          <button class="feedback-btn" data-type="interested">Interested</button>
          <button class="feedback-btn" data-type="watched">Watched</button>
        </div>
      `;
			container.appendChild(div);

			const [interestedBtn, watchedBtn] = div.querySelectorAll(".feedback-btn");

			const handleToggle = async (button, type) => {
				const isActive = button.classList.contains("active");
				if (isActive) {
					console.log(`✉️ Removing feedback: ${type} for movie ${m.movieId}`);
					button.classList.remove("active");
					button.textContent = type.charAt(0).toUpperCase() + type.slice(1);
					await sendFeedback(m.movieId, type, false);
				} else {
					// Deactivate other button
					const otherBtn = type === "interested" ? watchedBtn : interestedBtn;
					otherBtn.classList.remove("active");
					otherBtn.textContent =
						otherBtn.dataset.type.charAt(0).toUpperCase() +
						otherBtn.dataset.type.slice(1);

					console.log(`✉️ Sending feedback: ${type} for movie ${m.movieId}`);
					button.classList.add("active");
					button.textContent =
						type === "interested" ? "Marked Interested" : "Marked Watched";
					await sendFeedback(m.movieId, type, true);
				}
				await loadHistory();
			};

			interestedBtn.addEventListener("click", () =>
				handleToggle(interestedBtn, "interested")
			);
			watchedBtn.addEventListener("click", () =>
				handleToggle(watchedBtn, "watched")
			);
		});
	} catch (e) {
		console.error("❌ Failed to load recommendations:", e);
		container.innerHTML = "<em>Failed to load recommendations.</em>";
	} finally {
		showSpinner(false);
	}
});

// ===== Send Feedback =====
async function sendFeedback(movieId, type, add = true) {
	if (!currentUsername) return alert("Please log in first!");
	showSpinner(true);

	try {
		await fetch(`${API_BASE_URL}/feedback`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				username: currentUsername,
				movie_id: parseInt(movieId),
				interaction: add ? type : "remove",
			}),
		});
	} catch (e) {
		console.error("❌ Feedback error:", e);
		alert("Failed to update feedback.");
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
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		const div = document.getElementById("history");
		div.innerHTML = "";

		if (!data.history.length) {
			div.innerHTML = "<em>No history yet.</em>";
			return;
		}

		data.history.forEach((h) => {
			const item = document.createElement("div");
			item.className = "movie-card";
			item.innerHTML = `
        <span>${h.title}</span><br>
        <em>${h.interaction}</em><br>
        <em>Genres: ${h.genres.join(", ")}</em>
      `;
			div.appendChild(item);
		});
	} catch (e) {
		console.warn("⚠️ Failed to load history:", e);
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
