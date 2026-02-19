#!/usr/bin/env bash
# Interactive example runner for scry-engine, scry-chart, and scry-learn.
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
    "scatter3d||"
    "dual_y_demo||"
)

CAT2_NAME="Feature Demos"
CAT2_EXAMPLES=(
    "showcase||"
    "animation_demo||"
    "spring_sequence_demo||"
    "chart_integration||"
    "ml_3d_viz||"
    "pixel_dashboard||"
)

CAT3_NAME="Gallery — Generative Art"
CAT3_EXAMPLES=(
    "aurora_borealis||"
    "fractal_dreams||"
    "fluid_symphony||"
    "sacred_geometry||"
    "hypnotic_tunnels||"
    "wave_interference||"
    "particle_life||"
    "postmodern_manifesto||"
    "illusions||"
    "mind_benders||"
)

CAT4_NAME="SDF Scenes"
CAT4_EXAMPLES=(
    "sdf_showcase|sdf|"
    "masonic_mirror|sdf-gpu,widget|"
    "obsidian_mirror|sdf|"
    "text3d_showcase|sdf-text|"
)

CAT5_NAME="Animated Showcases"
CAT5_EXAMPLES=(
    "circus_ball|text,widget|"
    "fastfetch_anim||"
    "mission_control||"
    "line_drawing|svg,widget|"
)

CAT6_NAME="Charts (scry-chart)"
CAT6_EXAMPLES=(
    "demo||scry-chart"
    "showcase||scry-chart"
    "scatter_demo||scry-chart"
    "dashboard||scry-chart"
    "subplot_demo||scry-chart"
    "interactive||scry-chart"
    "advanced_charts||scry-chart"
    "render_png||scry-chart"
    "render_all||scry-chart"
    "chart_gallery||scry-chart"
    "font_scaling_demo||scry-chart"
)

CAT7_NAME="ML (scry-learn)"
CAT7_EXAMPLES=(
    "industry_report||scry-learn"
    "ml_viz_showcase|viz|scry-learn"
    "live_training|live-plot|scry-learn"
)

CAT8_NAME="Window Demos"
CAT8_EXAMPLES=(
    "window_demo|window|"
)

ALL_CATS=(1 2 3 4 5 6 7 8)

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

build_example() {
    parse_example "$1"
    local cmd="cargo build --example $EX_NAME --release"
    if [[ -n "$EX_FEATURES" ]]; then
        cmd+=" --features \"$EX_FEATURES\""
    fi
    if [[ -n "$EX_PACKAGE" ]]; then
        cmd+=" -p $EX_PACKAGE"
    fi

    printf "  %-28s" "$EX_NAME"
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${RESET}"
        return 0
    else
        echo -e "${RED}✗${RESET}"
        return 1
    fi
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
            printf "  ${YELLOW}%2d${RESET}) %-28s ${DIM}(%d examples)${RESET}\n" "$cat" "$name" "$count"
        done

        echo ""
        echo -e "  ${YELLOW} 0${RESET}) Run ALL examples"
        echo -e "  ${YELLOW} t${RESET}) Build-test ALL examples (no run)"
        echo -e "  ${YELLOW} q${RESET}) Quit"
        echo ""
        echo -n "  Select category: "
        read -r choice

        case "$choice" in
            q|Q)
                echo -e "\n${GREEN}Done.${RESET}"
                exit 0
                ;;
            t|T)
                echo ""
                echo -e "${CYAN}${BOLD}── Build-testing all examples ──${RESET}"
                echo ""
                local pass=0 fail=0
                for cat in "${ALL_CATS[@]}"; do
                    local var="CAT${cat}_EXAMPLES[@]"
                    local examples=("${!var}")
                    for ex in "${examples[@]}"; do
                        if build_example "$ex"; then
                            ((pass++))
                        else
                            ((fail++))
                        fi
                    done
                done
                echo ""
                echo -e "  ${GREEN}$pass passed${RESET}, ${RED}$fail failed${RESET}"
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
                if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#ALL_CATS[@]} )); then
                    show_category_menu "$choice"
                else
                    echo -e "  ${RED}Invalid choice${RESET}"
                fi
                ;;
        esac
    done
}

main_menu
