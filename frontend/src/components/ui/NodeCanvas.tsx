"use client";

import React, { useEffect, useRef } from "react";

/**
 * NodeCanvas – Circuit-board particle mesh.
 * Reduced particle count for professionalism:
 *  - Fewer, slower, smaller nodes (25 → was 40)
 *  - Lower opacity (0.15 → was 0.4)
 *  - Pure CSS fade transition keeps it buttery
 */
export function NodeCanvas() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        let animationFrameId: number;
        let particles: Particle[] = [];
        const PARTICLE_COUNT = 25;
        const CONNECTION_DISTANCE = 160;

        class Particle {
            x: number;
            y: number;
            size: number;
            speedX: number;
            speedY: number;

            constructor(width: number, height: number) {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.size = Math.random() * 1.5 + 0.5; // Smaller nodes
                this.speedX = (Math.random() - 0.5) * 0.3; // Slower drift
                this.speedY = (Math.random() - 0.5) * 0.3;
            }

            update(width: number, height: number) {
                this.x += this.speedX;
                this.y += this.speedY;

                if (this.x > width) this.x = 0;
                if (this.x < 0) this.x = width;
                if (this.y > height) this.y = 0;
                if (this.y < 0) this.y = height;
            }

            draw(ctx: CanvasRenderingContext2D) {
                // Draw node dot
                ctx.fillStyle = "rgba(14, 165, 233, 0.6)";
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();

                // Minimal cross-hair node indicator
                ctx.strokeStyle = "rgba(14, 165, 233, 0.15)";
                ctx.lineWidth = 0.5;
                ctx.strokeRect(this.x - 3, this.y - 3, 6, 6);
            }
        }

        const init = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            particles = [];
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                particles.push(new Particle(canvas.width, canvas.height));
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach((p, index) => {
                p.update(canvas.width, canvas.height);
                p.draw(ctx);

                // Orthogonal (PCB-trace) connections
                for (let j = index + 1; j < particles.length; j++) {
                    const dx = p.x - particles[j].x;
                    const dy = p.y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < CONNECTION_DISTANCE) {
                        const alpha = 0.12 * (1 - dist / CONNECTION_DISTANCE);
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(14, 165, 233, ${alpha})`;
                        ctx.lineWidth = 0.7;

                        const midX = p.x + (particles[j].x - p.x) / 2;
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(midX, p.y);
                        ctx.lineTo(midX, particles[j].y);
                        ctx.lineTo(particles[j].x, particles[j].y);

                        ctx.stroke();
                    }
                }
            });

            animationFrameId = requestAnimationFrame(animate);
        };

        init();
        animate();

        const handleResize = () => init();
        // Passive listener for performance (best-v: client-passive-event-listeners)
        window.addEventListener("resize", handleResize, { passive: true });

        return () => {
            cancelAnimationFrame(animationFrameId);
            window.removeEventListener("resize", handleResize);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 pointer-events-none z-0 opacity-20"
        />
    );
}
