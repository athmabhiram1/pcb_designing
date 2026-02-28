import React from "react";
import { cn } from "@/lib/utils";

interface StatusLabelProps {
    status: "critical" | "warning" | "success" | "info" | "neutral";
    children: React.ReactNode;
    className?: string;
}

const variants = {
    critical: "bg-red-500/10 text-red-500 border-red-500/20",
    warning: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
    success: "bg-green-500/10 text-green-500 border-green-500/20",
    info: "bg-primary/10 text-primary border-primary/20",
    neutral: "bg-white/5 text-white/50 border-white/10",
};

export function StatusLabel({ status, children, className }: StatusLabelProps) {
    return (
        <span
            className={cn(
                "px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border",
                variants[status],
                className
            )}
        >
            {children}
        </span>
    );
}
