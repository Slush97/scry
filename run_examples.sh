#!/usr/bin/env bash
# Interactive example runner for scry-engine and scry-chart.
# Select a category, then pick an example to run.

set -euo pipefail
cd "$(dirname "$0")"

# ── Colors ──────────────────────────────────────────────────────────
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
MAGENTA='\033[35m'
RED='\033[31m'
RESET='\033[0m'

# ── Example definitions ────────────────────────────────────────────
# Format: "name|features|package"
# - features: comma-separated cargo features (empty = none)
# - package: cargo -p flag (empty = root crate)

CAT1_NAME="Getting Started"
CAT1_EXAMPLES=(
    "simple_shapes||"
    "cube_3d||"
)

CAT2_NAME="Feature Demos"
CAT2_EXAMPLES=(
    "showcase||"
    "feature_showcase||"
    "new_features||"
    "animation_demo||"
    "spring_sequence_demo||"
    "chart_features||"
)

CAT3_NAME="Optical Illusions"
CAT3_EXAMPLES=(
    "illusions||"
    "mind_benders||"
)

CAT4_NAME="Generative Art"
CAT4_EXAMPLES=(
    "fractal_dreams||"
    "sacred_geometry||"
    "fluid_symphony||"
    "hypnotic_tunnels||"
    "aurora_borealis||"
    "wave_interference||"
    "particle_life||"
    "postmodern_manifesto||"
)

CAT5_NAME="Stress Tests"
CAT5_EXAMPLES=(
    "powertest||"
)

CAT6_NAME="SDF Scenes"
CAT6_EXAMPLES=(
    "sdf_showcase|sdf|"
    "masonic_mirror|sdf,text,widget|"
    "obsidian_mirror||"
)

CAT7_NAME="Animated Showcases"
CAT7_EXAMPLES=(
    "masonic_mirror|sdf,text,widget|"
    "circus_ball|text,widget|"
    "session3_showcase||"
    "fastfetch_anim||"
)

CAT8_NAME="Charts (scry-chart)"
CAT8_EXAMPLES=(
    "scatter_demo||scry-chart"
    "dashboard||scry-chart"
    "demo||scry-chart"
    "showcase||scry-chart"
    "chart_showcase||scry-chart"
    "interactive||scry-chart"
    "robustness_test||scry-chart"
    "feature_showcase||scry-chart"
    "tier2_charts||scry-chart"
    "subplot_demo||scry-chart"
    "visual_demo||scry-chart"
    "formatting_showcase||scry-chart"
)

CAT9_NAME="Window Demos"
CAT9_EXAMPLES=(
    "window_demo|window|"
)

CAT10_NAME="Other"
CAT10_EXAMPLES=(
    "line_drawing|svg,widget|"
    "dual_y_demo||"
    "pixel_dashboard||"
    "mission_control||"
    "ml_3d_demo||"
    "scatter3d||"
)

ALL_CATS=(1 2 3 4 5 6 7 8 9 10)

# ── Helper functions ───────────────────────────────────────────────

get_cat_name() {
    local var="CAT${1}_NAME"
    echo "${!var}"
}

get_cat_examples() {
    local var="CAT${1}_EXAMPLES[@]"
    echo "${!var}"
}

get_cat_count() {
    local var="CAT${1}_EXAMPLES[@]"
    local arr=("${!var}")
    echo "${#arr[@]}"
}

parse_example() {
    # Input: "name|features|package"
    IFS='|' read -r EX_NAME EX_FEATURES EX_PACKAGE <<< "$1"
}

run_example() {
    parse_example "$1"
    local cmd="cargo run --example $EX_NAME --release"
    if [[ -n "$EX_FEATURES" ]]; then
        cmd+=" --features \"$EX_FEATURES\""
    fi
    if [[ -n "$EX_PACKAGE" ]]; then
        cmd+=" -p $EX_PACKAGE"
    fi

    echo ""
    echo -e "${GREEN}>>> ${BOLD}$EX_NAME${RESET}"
    if [[ -n "$EX_FEATURES" ]]; then
        echo -e "    ${DIM}features: $EX_FEATURES${RESET}"
    fi
    if [[ -n "$EX_PACKAGE" ]]; then
        echo -e "    ${DIM}package: $EX_PACKAGE${RESET}"
    fi
    echo -e "    ${DIM}$cmd${RESET}"
    echo ""

    # Run with trap to catch Ctrl+C and return to menu
    eval "$cmd" || true
    echo ""
}

show_category_menu() {
    local cat_num="$1"
    local cat_name
    cat_name=$(get_cat_name "$cat_num")
    local var="CAT${cat_num}_EXAMPLES[@]"
    local examples=("${!var}")
    local count=${#examples[@]}

    while true; do
        echo ""
        echo -e "${CYAN}${BOLD}── $cat_name ($count examples) ──${RESET}"
        echo ""
        for i in "${!examples[@]}"; do
            parse_example "${examples[$i]}"
            local extra=""
            if [[ -n "$EX_FEATURES" ]]; then
                extra="${DIM} [${EX_FEATURES}]${RESET}"
            fi
            if [[ -n "$EX_PACKAGE" ]]; then
                extra+="${DIM} (-p ${EX_PACKAGE})${RESET}"
            fi
            echo -e "  ${YELLOW}$((i + 1))${RESET}) $EX_NAME$extra"
        done
        echo ""
        echo -e "  ${YELLOW}a${RESET}) Run all in sequence"
        echo -e "  ${YELLOW}b${RESET}) Back to main menu"
        echo ""
        echo -n "  Select: "
        read -r choice

        case "$choice" in
            a|A)
                for ex in "${examples[@]}"; do
                    run_example "$ex"
                done
                ;;
            b|B|"")
                return
                ;;
            *)
                if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= count )); then
                    run_example "${examples[$((choice - 1))]}"
                else
                    echo -e "  ${RED}Invalid choice${RESET}"
                fi
                ;;
        esac
    done
}

# ── Main menu ──────────────────────────────────────────────────────

main_menu() {
    while true; do
        echo ""
        echo -e "${MAGENTA}${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
        echo -e "${MAGENTA}${BOLD}║       scry — Interactive Example Runner                 ║${RESET}"
        echo -e "${MAGENTA}${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
        echo ""

        for cat in "${ALL_CATS[@]}"; do
            local name count
            name=$(get_cat_name "$cat")
            count=$(get_cat_count "$cat")
            printf "  ${YELLOW}%2d${RESET}) %-24s ${DIM}(%d examples)${RESET}\n" "$cat" "$name" "$count"
        done

        echo ""
        echo -e "  ${YELLOW} 0${RESET}) Run ALL examples"
        echo -e "  ${YELLOW} q${RESET}) Quit"
        echo ""
        echo -n "  Select category: "
        read -r choice

        case "$choice" in
            q|Q)
                echo -e "\n${GREEN}Done.${RESET}"
                exit 0
                ;;
            0)
                for cat in "${ALL_CATS[@]}"; do
                    local var="CAT${cat}_EXAMPLES[@]"
                    local examples=("${!var}")
                    local cat_name
                    cat_name=$(get_cat_name "$cat")
                    echo -e "\n${CYAN}${BOLD}── $cat_name ──${RESET}"
                    for ex in "${examples[@]}"; do
                        run_example "$ex"
                    done
                done
                ;;
            *)
                if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= 10 )); then
                    show_category_menu "$choice"
                else
                    echo -e "  ${RED}Invalid choice${RESET}"
                fi
                ;;
        esac
    done
}

main_menu
