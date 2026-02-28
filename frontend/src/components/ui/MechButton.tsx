"use client";

import React from "react";
import { motion, HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface MechButtonProps extends HTMLMotionProps<"button"> {
    variant?: "primary" | "secondary" | "outline" | "ghost";
    size?: "sm" | "md" | "lg";
    children: React.ReactNode;
}

export const MechButton = React.forwardRef<HTMLButtonElement, MechButtonProps>(
    ({ className, variant = "primary", size = "md", children, ...props }, ref) => {
        const variants = {
            primary: "bg-primary/10 border-primary/30 text-primary hover:bg-primary/20 hover:border-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]",
            secondary: "bg-secondary/10 border-secondary/30 text-secondary hover:bg-secondary/20 hover:border-secondary",
            outline: "bg-transparent border-white/10 text-white/70 hover:bg-white/5 hover:text-white hover:border-white/30",
            ghost: "bg-transparent border-transparent text-white/50 hover:text-white hover:bg-white/5",
        };

        const sizes = {
            sm: "px-3 py-1.5 text-xs font-mono tracking-tight",
            md: "px-5 py-2.5 text-sm font-display tracking-wide",
            lg: "px-8 py-4 text-base font-display font-medium tracking-wider uppercase",
        };

        return (
            <motion.button
                ref={ref}
                whileHover={{ translateY: -1 }}
                whileTap={{ translateY: 1, scale: 0.98 }}
                className={cn(
                    "relative border transition-all duration-100 flex items-center justify-center gap-2 group overflow-hidden",
                    variants[variant],
                    sizes[size],
                    className
                )}
                {...props}
            >
                {/* Scan line effect on hover */}
                <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-100 overflow-hidden">
                    <div className="scan-line" />
                </div>

                {/* Corner accents */}
                <div className="absolute top-0 left-0 w-1 h-1 border-t border-l border-current opacity-40 group-hover:opacity-100" />
                <div className="absolute bottom-0 right-0 w-1 h-1 border-b border-r border-current opacity-40 group-hover:opacity-100" />

                <span className="relative z-10">{children}</span>
            </motion.button>
        );
    }
);

MechButton.displayName = "MechButton";
