from typing import Optional, Tuple, List
from game.logic.base import BaseLogic
from game.models import Board, GameObject, Position
from game.util import get_direction
import math

class MultiWeightedGreedyLogic(BaseLogic):
    def __init__(self) -> None:
        self.goal_position: Optional[Position] = None
        self.static_goals: list[Position] = []
        self.static_goal_teleport: GameObject = None
        self.path_cache: dict = {}
        self.last_board_hash: str = ""
        self.previous_position: Optional[Position] = None  # Anti-stuck mechanism
        self.stuck_counter: int = 0
        
        # DHG Parameters - Fine-tuned values
        self.SAFETY_WEIGHT = 2.5
        self.CLUSTER_WEIGHT = 2.0
        self.TIME_PRESSURE_THRESHOLD = 15.0
        self.DANGER_RADIUS = 4
        self.TACKLE_OPPORTUNITY_WEIGHT = 3.0
        
        # Additional optimization parameters
        self.MIN_DIAMONDS_FOR_RED = 3
        self.BASE_RETURN_BUFFER = 3
        self.DIAMOND_CLUSTER_RADIUS = 3
        self.ENDGAME_TIME_THRESHOLD = 30.0  # Switch to endgame strategy
        self.COMPETITION_AWARENESS_FACTOR = 1.5  # Competitive multiplier

    def get_board_hash(self, board: Board) -> str:
        """Create a simple hash of board state for caching"""
        diamond_positions = sorted([(d.position.x, d.position.y, d.properties.points) for d in board.diamonds])
        bot_positions = sorted([(b.position.x, b.position.y, b.properties.diamonds) for b in board.bots])
        return str(hash((tuple(diamond_positions), tuple(bot_positions))))

    def is_endgame(self, current_bot: GameObject) -> bool:
        """Determine if we're in endgame phase"""
        time_left = current_bot.properties.milliseconds_left / 1000.0
        return time_left <= self.ENDGAME_TIME_THRESHOLD

    def calculate_competitive_pressure(self, board: Board, current_bot: GameObject) -> float:
        """Calculate how competitive the game situation is"""
        scores = [bot.properties.score for bot in board.bots]
        if len(scores) <= 1:
            return 1.0
            
        scores.sort(reverse=True)
        leader_score = scores[0]
        our_score = current_bot.properties.score
        
        if leader_score == 0:
            return 1.0
            
        # Higher pressure if we're behind
        score_ratio = our_score / leader_score
        return max(0.5, 2.0 - score_ratio)  # Range: 0.5 to 1.5

    def calculate_tackle_opportunity(self, position: Position, board: Board, 
                                   current_bot: GameObject) -> float:
        """Enhanced tackle opportunity calculation"""
        tackle_value = 0
        
        for bot in board.bots:
            if bot.id == current_bot.id or bot.properties.diamonds == 0:
                continue
                
            distance = abs(bot.position.x - position.x) + abs(bot.position.y - position.y)
            
            # Enhanced tackle conditions
            if distance <= 2:
                base_distance = abs(bot.properties.base.x - bot.position.x) + abs(bot.properties.base.y - bot.position.y)
                
                # Predict enemy next move
                enemy_will_move_to_base = (bot.properties.diamonds >= 3 and base_distance <= 2)
                
                if not enemy_will_move_to_base and bot.properties.diamonds >= 2 and base_distance > 3:
                    # Higher value for intercept opportunities
                    intercept_bonus = 1.0
                    if distance == 1:  # Can tackle next turn
                        intercept_bonus = 2.0
                        
                    tackle_value += bot.properties.diamonds * (3 - distance) * intercept_bonus
                    
        return tackle_value

    def calculate_safety_score(self, position: Position, board: Board, 
                              current_bot: GameObject) -> float:
        """Enhanced safety calculation with multi-enemy threat assessment"""
        safety = 1.0
        total_threat = 0
        
        for bot in board.bots:
            if bot.id == current_bot.id:
                continue
                
            distance = abs(bot.position.x - position.x) + abs(bot.position.y - position.y)
            
            if distance <= self.DANGER_RADIUS:
                enemy_to_us = distance
                enemy_to_base = abs(bot.properties.base.x - bot.position.x) + abs(bot.properties.base.y - bot.position.y)
                
                # Dynamic threat calculation
                base_threat = 1.0 + (bot.properties.diamonds * 0.2)  # More diamonds = more aggressive
                
                # Reduce threat if enemy is returning to base
                if bot.properties.diamonds >= 3 and enemy_to_base :
                    base_threat *= 0.4
                
                # Distance-based threat decay
                distance_factor = (self.DANGER_RADIUS - distance) / self.DANGER_RADIUS
                threat_level = base_threat * distance_factor
                total_threat += threat_level
        
        # Apply cumulative threat with diminishing returns
        safety = 1.0 / (1.0 + total_threat * 0.3)
        return max(safety, 0.02)  # Very low minimum for high-risk situations

    def calculate_enhanced_cluster_weight(self, position: Position, board: Board, 
                                        current_bot: GameObject) -> float:
        """Enhanced cluster calculation with competitive analysis"""
        weight = 0
        red_diamond_bonus = 0
        competitive_multiplier = self.calculate_competitive_pressure(board, current_bot)
        
        # Count red diamonds in cluster first
        red_clusters = 0
        for diamond in board.diamonds:
            if (diamond.properties.points == 2 and
                abs(diamond.position.x - position.x) <= self.DIAMOND_CLUSTER_RADIUS and
                abs(diamond.position.y - position.y) <= self.DIAMOND_CLUSTER_RADIUS):
                red_clusters += 1

        for diamond in board.diamonds:
            distance = abs(diamond.position.x - position.x) + abs(diamond.position.y - position.y)
            if distance <= 5:  # Extended cluster detection
                # Steeper decay for distant diamonds
                decay_factor = math.exp(-distance / 1.5)
                base_value = diamond.properties.points * decay_factor
                
                # Enhanced red diamond bonus
                if diamond.properties.points == 2:
                    red_diamond_bonus += base_value * (0.7 + red_clusters * 0.1)
                
                weight += base_value
        
        total_weight = (weight + red_diamond_bonus) * competitive_multiplier
        return total_weight

    def calculate_path_efficiency(self, start: Position, target: Position, board: Board) -> float:
        """Enhanced path efficiency with obstacle avoidance"""
        direct_distance = abs(target.x - start.x) + abs(target.y - start.y)
        
        # Check teleporter efficiency
        best_teleporter_distance = float('inf')
        teleporter_bonus = 1.0
        
        for teleporter in board.game_objects:
            if teleporter.type != "TeleportGameObject":
                continue
                
            other_teleport = self.find_other_teleport(teleporter, board)
            if not other_teleport:
                continue
                
            teleporter_path = (abs(teleporter.position.x - start.x) + 
                             abs(teleporter.position.y - start.y) +
                             abs(other_teleport.position.x - target.x) + 
                             abs(other_teleport.position.y - target.y))
            
            if teleporter_path < best_teleporter_distance:
                best_teleporter_distance = teleporter_path
                # Bonus for teleporter usage when it's significantly better
                if teleporter_path < direct_distance * 0.7:
                    teleporter_bonus = 1.3

        actual_distance = min(direct_distance, best_teleporter_distance)
        efficiency = (direct_distance / max(actual_distance, 1)) * teleporter_bonus
        return min(efficiency, 2.0)  # Cap efficiency bonus

    def calculate_dhg_score(self, current: Position, target: Position, 
                           target_value: int, board: Board, 
                           current_bot: GameObject) -> float:
        """Enhanced DHG scoring with endgame awareness"""
        
        manhattan_distance = abs(target.x - current.x) + abs(target.y - current.y)
        if manhattan_distance == 0:
            return float('inf')
            
        # Core calculations
        cluster_weight = self.calculate_enhanced_cluster_weight(target, board, current_bot)
        safety_score = self.calculate_safety_score(target, board, current_bot)
        tackle_opportunity = self.calculate_tackle_opportunity(target, board, current_bot)
        path_efficiency = self.calculate_path_efficiency(current, target, board)
        
        # Enhanced time pressure
        time_left = current_bot.properties.milliseconds_left / 1000.0
        if self.is_endgame(current_bot):
            # Endgame strategy: prioritize closer, safer targets
            time_pressure = 2.0 + (self.ENDGAME_TIME_THRESHOLD - time_left) / 10.0
            safety_score *= 1.5  # Extra safety in endgame
        elif time_left < self.TIME_PRESSURE_THRESHOLD:
            time_pressure = 1.0 + (self.TIME_PRESSURE_THRESHOLD - time_left) / self.TIME_PRESSURE_THRESHOLD
        else:
            time_pressure = 1.0
            
        # Dynamic inventory consideration
        inventory_factor = 1.0
        if current_bot.properties.diamonds >= 3:
            # Strongly prefer red diamonds when inventory is getting full
            inventory_factor = 1.0 + (target_value - 1) * 0.8
        elif current_bot.properties.diamonds <= 1:
            # Early game: be more flexible
            inventory_factor = 1.0 + (target_value - 1) * 0.2
            
        # Competitive factor
        competitive_pressure = self.calculate_competitive_pressure(board, current_bot)
        
        # Enhanced DHG Score calculation
        value_factor = (target_value * inventory_factor + 
                       cluster_weight * self.CLUSTER_WEIGHT + 
                       tackle_opportunity * self.TACKLE_OPPORTUNITY_WEIGHT) * competitive_pressure
        
        distance_factor = manhattan_distance * time_pressure / path_efficiency
        
        dhg_score = (value_factor * safety_score * self.SAFETY_WEIGHT) / distance_factor
        
        return dhg_score

    def should_return_to_base_enhanced(self, board_bot: GameObject, board: Board, 
                                     best_diamond_distance: float) -> bool:
        """Enhanced base return with endgame and competitive logic"""
        base = board_bot.properties.base
        base_distance = abs(base.x - board_bot.position.x) + abs(base.y - board_bot.position.y)
        time_left = board_bot.properties.milliseconds_left / 1000.0
        
        # Critical conditions
        if board_bot.properties.diamonds >= 5:
            return True
            
        # Enhanced time management
        time_buffer = 3 if not self.is_endgame(board_bot) else 5  # More conservative in endgame
        if base_distance >= time_left - time_buffer:
            return True
            
        # Endgame strategy: secure diamonds earlier
        if self.is_endgame(board_bot) and board_bot.properties.diamonds >= 2:
            return True
            
        # Dynamic inventory thresholds
        if board_bot.properties.diamonds >= 4:
            return True
            
        # Enhanced risk assessment
        enemies_nearby = 0
        enemy_threat_level = 0
        high_value_enemies = 0
        
        for bot in board.bots:
            if bot.id == board_bot.id:
                continue
            distance = abs(bot.position.x - board_bot.position.x) + abs(bot.position.y - board_bot.position.y)
            if distance <= 4:  # Extended threat detection
                enemies_nearby += 1
                threat_weight = (5 - distance) * (1 + bot.properties.diamonds * 0.3)
                enemy_threat_level += threat_weight
                
                if bot.properties.diamonds >= 3:
                    high_value_enemies += 1
        
        # Return if surrounded or high threat
        if enemy_threat_level > 6 and board_bot.properties.diamonds >= 2:
            return True
            
        if high_value_enemies >= 2 and board_bot.properties.diamonds >= 1:
            return True
            
        # Competitive opportunity cost
        competitive_factor = self.calculate_competitive_pressure(board, board_bot)
        distance_threshold = 0.9 if competitive_factor > 1.2 else 0.8
        
        if (board_bot.properties.diamonds >= 3 and 
            base_distance < best_diamond_distance * distance_threshold):
            return True
            
        # Scarcity-based return with competitive awareness
        diamond_scarcity_threshold = 4 if competitive_factor > 1.3 else 3
        if len(board.diamonds) <= diamond_scarcity_threshold and board_bot.properties.diamonds >= 2:
            return True
            
        return False

    def evaluate_red_button_strategy(self, board: Board, current_bot: GameObject) -> float:
        """Enhanced red button strategy with timing optimization"""
        if not board.game_objects:
            return 0
            
        red_button = None
        for obj in board.game_objects:
            if obj.type == "DiamondButtonGameObject":
                red_button = obj
                break
                
        if not red_button:
            return 0
            
        total_diamonds = len(board.diamonds)
        red_diamonds = sum(1 for d in board.diamonds if d.properties.points == 2)
        time_left = current_bot.properties.milliseconds_left / 1000.0
        
        # Enhanced competitive analysis
        competitive_pressure = self.calculate_competitive_pressure(board, current_bot)
        current_ranking = self.get_score_ranking(board, current_bot)
        
        # Scarcity factor with competitive weighting
        scarcity_factor = max(0, (12 - total_diamonds) / 12.0) * competitive_pressure
        
        # Red diamond ratio with threshold adjustment
        red_ratio = red_diamonds / max(total_diamonds, 1)
        optimal_red_ratio = 0.35  # Target ratio
        red_ratio_factor = max(0, (optimal_red_ratio - red_ratio) / optimal_red_ratio)
        
        # Ranking-based desperation
        total_bots = len(board.bots)
        ranking_pressure = (current_ranking - 1) / max(total_bots - 1, 1)
        
        # Timing optimization
        if time_left < 25:  # Too late
            time_factor = 0
        elif time_left > 100:  # Too early
            time_factor = 0.4
        elif 60 <= time_left <= 90:  # Optimal timing
            time_factor = 1.2
        else:
            time_factor = 0.8
            
        # Distance consideration
        button_distance = abs(red_button.position.x - current_bot.position.x) + abs(red_button.position.y - current_bot.position.y)
        distance_factor = max(0.3, 1.0 - button_distance / 10.0)
        
        red_button_score = (scarcity_factor * 0.25 + 
                           red_ratio_factor * 0.25 +
                           ranking_pressure * 0.3 + 
                           time_factor * 0.2) * 18 * distance_factor
        
        return red_button_score

    def get_score_ranking(self, board: Board, current_bot: GameObject) -> int:
        """Get current bot's ranking by score"""
        scores = [(bot.properties.score, bot.id) for bot in board.bots]
        scores.sort(reverse=True)
        
        for i, (score, bot_id) in enumerate(scores):
            if bot_id == current_bot.id:
                return i + 1
        return len(board.bots)

    def find_optimal_target(self, board_bot: GameObject, board: Board) -> Tuple[Position, float]:
        """Enhanced target finding with anti-stuck mechanism"""
        current = board_bot.position
        best_score = 0
        best_target = None
        best_distance = float('inf')
        
        # Anti-stuck mechanism
        if self.previous_position and self.previous_position == current:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.previous_position = current
        
        # If stuck, add randomness to break cycles
        stuck_penalty = self.stuck_counter * 0.1 if self.stuck_counter > 3 else 0
        
        # Enhanced diamond evaluation
        for diamond in board.diamonds:
            # Inventory constraints with endgame flexibility
            if self.is_endgame(board_bot):
                # In endgame, be more flexible about red diamonds
                if board_bot.properties.diamonds >= 4 and diamond.properties.points == 2:
                    continue
            else:
                if board_bot.properties.diamonds >= 4 and diamond.properties.points == 2:
                    continue
                    
            # Time-distance feasibility
            distance = abs(diamond.position.x - current.x) + abs(diamond.position.y - current.y)
            time_left = board_bot.properties.milliseconds_left / 1000.0
            
            required_time = distance * 2 + 2  # Buffer for base return
            if required_time > time_left:
                continue
                
            dhg_score = self.calculate_dhg_score(
                current, diamond.position, diamond.properties.points,
                board, board_bot
            )
            
            # Apply stuck penalty to previously targeted positions
            if self.stuck_counter > 3:
                dhg_score *= (1 - stuck_penalty)
            
            if dhg_score > best_score:
                best_score = dhg_score
                best_target = diamond.position
                best_distance = distance

        # Enhanced teleporter evaluation (keeping your existing logic)
        for teleporter in board.game_objects:
            if teleporter.type != "TeleportGameObject":
                continue
                
            other_teleport = self.find_other_teleport(teleporter, board)
            if not other_teleport:
                continue
                
            teleport_distance = (abs(teleporter.position.x - current.x) + 
                               abs(teleporter.position.y - current.y))
            
            accessible_diamonds = []
            for diamond in board.diamonds:
                if board_bot.properties.diamonds >= 4 and diamond.properties.points == 2:
                    continue
                    
                post_teleport_distance = (abs(other_teleport.position.x - diamond.position.x) + 
                                        abs(other_teleport.position.y - diamond.position.y))
                total_distance = teleport_distance + post_teleport_distance
                
                if total_distance <= 10:  # Slightly more lenient
                    accessible_diamonds.append((diamond, total_distance))
            
            if accessible_diamonds:
                accessible_diamonds.sort(key=lambda x: x[1])
                best_diamond, total_distance = accessible_diamonds[0]
                
                dhg_score = self.calculate_dhg_score(
                    other_teleport.position, best_diamond.position, 
                    best_diamond.properties.points, board, board_bot
                ) * 0.9  # Reduced penalty
                
                if dhg_score > best_score:
                    best_score = dhg_score
                    best_target = teleporter.position
                    best_distance = total_distance
                    self.static_goal_teleport = teleporter

        # Enhanced red button evaluation
        red_button_score = self.evaluate_red_button_strategy(board, board_bot)
        if red_button_score > best_score:
            for obj in board.game_objects:
                if obj.type == "DiamondButtonGameObject":
                    distance = (abs(obj.position.x - current.x) + 
                              abs(obj.position.y - current.y))
                    if distance <= 8:  # More lenient distance check
                        best_target = obj.position
                        best_distance = distance
                        break

        return best_target, best_distance

    def find_other_teleport(self, teleporter: GameObject, board: Board) -> GameObject:
        """Find the paired teleporter"""
        for obj in board.game_objects:
            if (obj.type == "TeleportGameObject" and 
                obj.properties.pair_id == teleporter.properties.pair_id and
                obj.id != teleporter.id):
                return obj
        return None

    def next_move(self, board_bot: GameObject, board: Board):
        """Enhanced next move with comprehensive decision making"""
        
        # Reset goals if at base
        if board_bot.position == board_bot.properties.base:
            self.static_goals = []
            self.static_goal_teleport = None
            self.path_cache.clear()
            
        # Find optimal target using enhanced DHG
        best_target, best_distance = self.find_optimal_target(board_bot, board)
        
        # Enhanced decision making
        if not best_target:
            self.goal_position = board_bot.properties.base
        elif self.should_return_to_base_enhanced(board_bot, board, best_distance):
            self.goal_position = board_bot.properties.base
        else:
            self.goal_position = best_target
    
        # Calculate movement direction
        dx, dy = get_direction(
            board_bot.position.x,
            board_bot.position.y,
            self.goal_position.x,
            self.goal_position.y,
        )
        
        # Prevent invalid (0,0) moves
        if dx == 0 and dy == 0:
            # Try moving in a valid direction based on board position
            if board_bot.position.x < board.width // 2:
                return 1, 0  # Move right
            else:
                return -1, 0  # Move left
        
        # Prevent diagonal moves (when dx == dy)
        if abs(dx) == abs(dy):
            # Prioritize horizontal movement
            return dx, 0
        
        return dx, dy